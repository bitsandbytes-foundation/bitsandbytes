#!/usr/bin/env python
"""Parse JUnit XML test reports and post a consolidated summary to Slack.

Designed for the bitsandbytes nightly integration tests that run downstream
test suites (transformers, accelerate, peft) against the current bnb build.

Usage:
    # Dry-run (print to stdout, no Slack):
    python scripts/integration_test_report.py --reports-dir reports/

    # Post to Slack:
    python scripts/integration_test_report.py --reports-dir reports/ --slack-channel bnb-ci-nightly
"""

import argparse
from datetime import date
import glob
import os
import sys
from xml.etree import ElementTree


def parse_junit_xml(file_path):
    """Parse a JUnit XML file and return structured results."""
    tree = ElementTree.parse(file_path)
    root = tree.getroot()

    # Handle both <testsuites><testsuite>... and bare <testsuite>...
    if root.tag == "testsuites":
        suites = root.findall("testsuite")
    else:
        suites = [root]

    tests = 0
    passed = 0
    failed = 0
    skipped = 0
    errors = 0
    total_time = 0.0
    failures = []

    for suite in suites:
        tests += int(suite.get("tests", 0))
        skipped += int(suite.get("skipped", 0))
        errors += int(suite.get("errors", 0))
        failed += int(suite.get("failures", 0))
        total_time += float(suite.get("time", 0))

        for testcase in suite.findall("testcase"):
            failure = testcase.find("failure")
            error = testcase.find("error")
            if failure is not None:
                failures.append(
                    {
                        "test": f"{testcase.get('classname', '')}::{testcase.get('name', '')}",
                        "message": failure.get("message", ""),
                    }
                )
            elif error is not None:
                failures.append(
                    {
                        "test": f"{testcase.get('classname', '')}::{testcase.get('name', '')}",
                        "message": error.get("message", ""),
                    }
                )

    passed = tests - failed - skipped - errors

    return {
        "tests": tests,
        "passed": passed,
        "failed": failed + errors,
        "skipped": skipped,
        "time": total_time,
        "failures": failures,
    }


def format_duration(seconds):
    """Format seconds into a human-readable string."""
    m, s = divmod(int(seconds), 60)
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def consolidate_reports(reports_dir):
    """Find and parse all JUnit XML files in the reports directory."""
    xml_files = sorted(glob.glob(os.path.join(reports_dir, "**", "*.xml"), recursive=True))

    if not xml_files:
        print(f"No XML report files found in {reports_dir}", file=sys.stderr)
        return {}

    results = {}
    for xml_file in xml_files:
        # Derive suite name from filename: "transformers.xml" -> "transformers"
        suite_name = os.path.splitext(os.path.basename(xml_file))[0]
        results[suite_name] = parse_junit_xml(xml_file)

    return results


def _success_rate(r):
    """Success rate: passed / (passed + failed), ignoring skipped."""
    run = r["passed"] + r["failed"]
    return (r["passed"] / run) if run > 0 else 1.0


def generate_markdown(results):
    """Generate a markdown summary report."""
    if not results:
        return "No test results found."

    total_passed = sum(r["passed"] for r in results.values())
    total_failed = sum(r["failed"] for r in results.values())
    total_skipped = sum(r["skipped"] for r in results.values())
    total_time = sum(r["time"] for r in results.values())

    lines = []
    lines.append("# BNB Integration Test Report")
    lines.append("")

    total_run = total_passed + total_failed
    if total_failed == 0:
        lines.append(f"All {total_run} tests passed in {format_duration(total_time)}.")
    else:
        lines.append(f"**{total_failed} failures** out of {total_run} tests in {format_duration(total_time)}.")
    if total_skipped > 0:
        lines.append(f"({total_skipped} skipped)")

    lines.append("")
    lines.append("| Suite | Tests | Passed | Failed | Skipped | Duration | Success Rate |")
    lines.append("|-------|------:|-------:|-------:|--------:|---------:|-------------:|")

    # Sort by success rate ascending (worst first)
    sorted_results = sorted(results.items(), key=lambda x: _success_rate(x[1]))

    for suite_name, r in sorted_results:
        run = r["passed"] + r["failed"]
        rate = f"{r['passed'] / run * 100:.1f}%" if run > 0 else "N/A"
        lines.append(
            f"| {suite_name} | {r['tests']} | {r['passed']} | {r['failed']} "
            f"| {r['skipped']} | {format_duration(r['time'])} | {rate} |"
        )

    # Failure details
    any_failures = any(r["failures"] for r in results.values())
    if any_failures:
        lines.append("")
        lines.append("## Failures")
        for suite_name, r in sorted_results:
            if r["failures"]:
                lines.append(f"### {suite_name}")
                lines.append("```")
                for f in r["failures"]:
                    if f["message"]:
                        lines.append(f"FAILED {f['test']} - {f['message']}")
                    else:
                        lines.append(f"FAILED {f['test']}")
                lines.append("```")
                lines.append("")

    return "\n".join(lines)


def create_slack_payload(results):
    """Create Slack Block Kit payload from results."""
    total_passed = sum(r["passed"] for r in results.values())
    total_failed = sum(r["failed"] for r in results.values())
    total_skipped = sum(r["skipped"] for r in results.values())

    total_run = total_passed + total_failed

    if total_run == 0:
        emoji = "⚠️"
        rate_str = "N/A"
    elif total_failed == 0:
        emoji = "✅"
        rate_str = "100%"
    elif total_failed / total_run < 0.1:
        emoji = "⚠️"
        rate_str = f"{total_passed / total_run * 100:.1f}%"
    else:
        emoji = "❌"
        rate_str = f"{total_passed / total_run * 100:.1f}%"

    summary = f"{emoji} *BNB Integration Tests:* {rate_str} success ({total_passed}/{total_run} tests"
    if total_skipped > 0:
        summary += f", {total_skipped} skipped"
    if total_failed > 0:
        summary += f", {total_failed} failed"
    summary += ")"

    # Build table — sorted by success rate ascending (worst first)
    sorted_results = sorted(results.items(), key=lambda x: _success_rate(x[1]))

    table_lines = ["```"]
    header = f"{'Suite':<15} {'Tests':>6} {'Failed':>7} {'Duration':>10} {'Success':>8}"
    table_lines.append(header)
    table_lines.append("-" * len(header))

    for suite_name, r in sorted_results:
        run = r["passed"] + r["failed"]
        rate = f"{r['passed'] / run * 100:.1f}%" if run > 0 else "N/A"
        table_lines.append(f"{suite_name:<15} {run:>6} {r['failed']:>7} {format_duration(r['time']):>10} {rate:>8}")

    table_lines.append("```")

    payload = [
        {"type": "section", "text": {"type": "mrkdwn", "text": summary}},
        {"type": "section", "text": {"type": "mrkdwn", "text": "\n".join(table_lines)}},
    ]

    # GitHub Actions link
    run_id = os.environ.get("GITHUB_RUN_ID")
    repo = os.environ.get("GITHUB_REPOSITORY", "bitsandbytes-foundation/bitsandbytes")
    if run_id:
        payload.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*<https://github.com/{repo}/actions/runs/{run_id}|View full report on GitHub>*",
                },
            }
        )

    payload.append(
        {
            "type": "context",
            "elements": [{"type": "plain_text", "text": f"Nightly integration test results for {date.today()}"}],
        }
    )

    return payload


def create_failure_thread_payloads(results):
    """Create per-suite Slack thread replies for failures."""
    threads = []

    for suite_name, r in results.items():
        if not r["failures"]:
            continue

        run = r["passed"] + r["failed"]
        rate = f"{r['passed'] / run * 100:.1f}%" if run > 0 else "N/A"
        lines = [f"*{suite_name}* (Success Rate: {rate})"]
        lines.append("```")
        for f in r["failures"]:
            if f["message"]:
                lines.append(f"FAILED {f['test']}")
                lines.append(f"  {f['message'][:200]}")
            else:
                lines.append(f"FAILED {f['test']}")
        lines.append("```")

        threads.append("\n".join(lines))

    return threads


def post_to_slack(channel, payload, thread_payloads):
    """Post the report to Slack."""
    from slack_sdk import WebClient

    token = os.environ.get("SLACK_API_TOKEN")
    if not token:
        print("SLACK_API_TOKEN not set, skipping Slack post", file=sys.stderr)
        return

    client = WebClient(token=token)

    # Main message
    response = client.chat_postMessage(
        channel=f"#{channel}",
        text="BNB Integration Test Results",
        blocks=payload,
    )
    print(f"Posted to #{channel}")

    # Threaded failure details
    ts = response["ts"]
    for thread_msg in thread_payloads:
        client.chat_postMessage(
            channel=f"#{channel}",
            thread_ts=ts,
            text=thread_msg,
        )

    if thread_payloads:
        print(f"Posted {len(thread_payloads)} failure thread replies")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reports-dir", default="reports", help="Directory containing JUnit XML files")
    parser.add_argument("--slack-channel", default=None, help="Slack channel name (omit to skip Slack)")
    parser.add_argument("--output", default=None, help="Write markdown report to file")
    parser.add_argument("--dry-run", action="store_true", help="Print Slack payload as JSON instead of posting")
    args = parser.parse_args()

    results = consolidate_reports(args.reports_dir)
    if not results:
        sys.exit(1)

    # Markdown report
    markdown = generate_markdown(results)

    if args.output:
        with open(args.output, "w") as f:
            f.write(markdown)
        print(f"Report written to {args.output}")

    # Always print markdown (for $GITHUB_STEP_SUMMARY piping)
    print(markdown)

    # Slack
    payload = create_slack_payload(results)
    thread_payloads = create_failure_thread_payloads(results)

    if args.dry_run:
        import json

        print("\n--- Slack main payload ---")
        print(json.dumps(payload, indent=2))
        for i, tp in enumerate(thread_payloads):
            print(f"\n--- Thread reply {i + 1} ---")
            print(tp)
    elif args.slack_channel:
        post_to_slack(args.slack_channel, payload, thread_payloads)


if __name__ == "__main__":
    main()
