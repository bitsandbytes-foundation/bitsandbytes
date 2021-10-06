#pragma once

#include "Portable.h"

namespace BinSearch {
namespace Details {

template <typename T>
bool isAligned(const T *p, size_t A)
{
    return (reinterpret_cast<size_t>(p) % A) == 0;
}

template <class T, size_t A=64>
struct AlignedVec
{
    AlignedVec()
        : m_storage(0)
        , m_data(0)
        , m_sz(0)
    {
    }

    static size_t nBytes(size_t sz)
    {
        return sz * sizeof(T) + A;
    }

    static size_t shiftAmt(char *p)
    {
        return A>1? (A - (reinterpret_cast<size_t>(p) % A)) % A: 0;
    }

    void setPtr(char *p, size_t sz)
    {
        m_sz = sz;
        m_data = reinterpret_cast<T *>(p + shiftAmt(p));
    }

    //void setPtr(T *p, size_t sz)
    //{
    //    m_sz = sz;
    //    if (A>1)
    //        myassert(((reinterpret_cast<size_t>(p) % A) == 0), "bad alignment");
    //    m_data = p;
    //}

    // internal allocation
    void resize(size_t sz)
    {
        m_storage = new char[nBytes(sz)];
        setPtr(m_storage, sz);
    }

    // external allocation
    void set(char *storage, size_t sz)
    {
        setPtr(storage, sz);
    }

    ~AlignedVec()
    {
        if (m_storage)
            delete [] m_storage;
    }

    size_t size() const { return m_sz; }
    T& operator[](size_t i) { return m_data[i]; }
    const T& operator[](size_t i) const { return m_data[i]; }
    T* begin()  { return m_data;  }
    T* end()  { return m_data+m_sz; }
    const T* begin() const { return m_data;  }
    const T* end() const { return m_data+m_sz; }
    T& front() { return m_data[0]; }
    T& back() { return m_data[m_sz-1]; }
    const T& front() const { return m_data[0]; }
    const T& back() const { return m_data[m_sz - 1]; }

private:
    char *m_storage;
    T *m_data;
    size_t m_sz;
};

} // namespace Details
} // namespace BinSearch
