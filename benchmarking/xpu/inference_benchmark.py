import argparse
import time

# import intel_extension_for_pytorch as ipex
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

MAX_NEW_TOKENS = 256

get_time = time.time

system_prompt = "You are a helpful assistant"
user_prompt = """Summarize this text please:

```Tell me, O muse, of that ingenious hero who travelled far and wide after he had sacked the famous town of Troy. Many cities did he visit, and many were the nations with whose manners and customs he was acquainted; moreover he suffered much by sea while trying to save his own life and bring his men safely home; but do what he might he could not save his men, for they perished through their own sheer folly in eating the cattle of the Sun-god Hyperion; so the god prevented them from ever reaching home. Tell me, too, about all these things, O daughter of Jove, from whatsoever source you may know them.

So now all who escaped death in battle or by shipwreck had got safely home except Ulysses, and he, though he was longing to return to his wife and country, was detained by the goddess Calypso, who had got him into a large cave and wanted to marry him. But as years went by, there came a time when the gods settled that he should go back to Ithaca; even then, however, when he was among his own people, his troubles were not yet over; nevertheless all the gods had now begun to pity him except Neptune, who still persecuted him without ceasing and would not let him get home.

Now Neptune had gone off to the Ethiopians, who are at the world's end, and lie in two halves, the one looking West and the other East. He had gone there to accept a hecatomb of sheep and oxen, and was enjoying himself at his festival; but the other gods met in the house of Olympian Jove, and the sire of gods and men spoke first. At that moment he was thinking of Aegisthus, who had been killed by Agamemnon's son Orestes; so he said to the other gods:

"See now, how men lay blame upon us gods for what is after all nothing but their own folly. Look at Aegisthus; he must needs make love to Agamemnon's wife unrighteously and then kill Agamemnon, though he knew it would be the death of him; for I sent Mercury to warn him not to do either of these things, inasmuch as Orestes would be sure to take his revenge when he grew up and wanted to return home. Mercury told him this in all good will but he would not listen, and now he has paid for everything in full."

Then Minerva said, "Father, son of Saturn, King of kings, it served Aegisthus right, and so it would any one else who does as he did; but Aegisthus is neither here nor there; it is for Ulysses that my heart bleeds, when I think of his sufferings in that lonely sea-girt island, far away, poor man, from all his friends. It is an island covered with forest, in the very middle of the sea, and a goddess lives there, daughter of the magician Atlas, who looks after the bottom of the ocean, and carries the great columns that keep heaven and earth asunder. This daughter of Atlas has got hold of poor unhappy Ulysses, and keeps trying by every kind of blandishment to make him forget his home, so that he is tired of life, and thinks of nothing but how he may once more see the smoke of his own chimneys. You, sir, take no heed of this, and yet when Ulysses was before Troy did he not propitiate you with many a burnt sacrifice? Why then should you keep on being so angry with him?"

And Jove said, "My child, what are you talking about? How can I forget Ulysses than whom there is no more capable man on earth, nor more liberal in his offerings to the immortal gods that live in heaven? Bear in mind, however, that Neptune is still furious with Ulysses for having blinded an eye of Polyphemus king of the Cyclopes. Polyphemus is son to Neptune by the nymph Thoosa, daughter to the sea-king Phorcys; therefore though he will not kill Ulysses outright, he torments him by preventing him from getting home. Still, let us lay our heads together and see how we can help him to return; Neptune will then be pacified, for if we are all of a mind he can hardly stand out against us."```"""

prompt = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt},
]


def get_inputs(tokenizer):
    inputs = tokenizer.apply_chat_template(
        prompt,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    return inputs


def get_streamer(tokenizer):
    streamer = Streamer(tokenizer)
    # streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    return streamer


class Streamer:
    def __init__(self, tokenizer, print_median=False):
        self.times = []
        self.print_median = print_median
        self.tokenizer = tokenizer

    def put(self, t):
        self.times.append(get_time())
        if len(self.times) > 1:
            print(f"Token latency: {1000 * (self.times[-1] - self.times[-2]):.1f} ms")

        if len(self.times) % 10 == 3 and self.print_median:
            ts = np.array(self.times)
            diff = ts[1:] - ts[:-1]
            # print("Token latency:", 1000 * diff, "ms")
            print("Token latency median:", np.median(1000 * diff), "ms")

    def print_report(self):
        times = np.array(self.times)
        diff = times[1:] - times[:-1]
        print(f"Median latency: {round(np.median(diff) * 1000, 2)}ms")
        percentiles = [10, 25, 50, 75, 90]
        print(
            "Latency percentiles",
            {p: round(1000 * float(np.percentile(diff, p)), 1) for p in percentiles},
        )

    def end(self, *args):
        pass


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run inference benchmark for LLM models")
    parser.add_argument(
        "--device",
        type=str,
        default="xpu",
        help="Device to run inference on (e.g., xpu, cuda, cpu)",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        help="Model ID from Hugging Face or local path",
    )
    parser.add_argument(
        "--attn",
        type=str,
        default="eager",
        choices=["eager", "flash_attention", "sdpa"],
        help="Attention implementation to use",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    device = args.device
    model_id = args.model_id

    print(f"Running inference on {device} with model {model_id}")
    print(f"Using attention implementation: {args.attn}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation=args.attn)

    inputs = get_inputs(tokenizer)
    streamer = get_streamer(tokenizer)

    inputs = inputs.to(device)
    model = model.to(device)

    generation_config = GenerationConfig(
        use_cache=True,
        forced_eos_token_id=1,
        eos_token_id=1,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
    )

    outputs = model.generate(
        **inputs,
        streamer=streamer,
        generation_config=generation_config,
    )

    # Print the final outputs (including the input prompt)
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(r"\Output (including prompt):")
    print("-" * 40)
    print(output_text)
    print("-" * 40)
    print(f"Peak memory usage: {torch.xpu.max_memory_allocated() / 1024**2:.0f}MB")

    streamer.print_report()
