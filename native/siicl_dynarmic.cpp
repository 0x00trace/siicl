// C interface to dynarmic for siicl. See siicl_dynarmic.h.

#include "siicl_dynarmic.h"

#include <sys/mman.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

#include <dynarmic/interface/A64/a64.h>
#include <dynarmic/interface/A64/config.h>
#include <dynarmic/interface/exclusive_monitor.h>

namespace {

constexpr uint64_t PAGE_BITS = 12;
constexpr uint64_t PAGE = uint64_t{1} << PAGE_BITS;

using Dynarmic::HaltReason;
using Dynarmic::A64::Exception;
using Dynarmic::A64::VAddr;
using Dynarmic::A64::Vector;

// sd_run's own halt reasons: a callback stopped the run, or sd_halt was called.
constexpr HaltReason STOPPED = HaltReason::UserDefined1;
constexpr HaltReason HALTED = HaltReason::UserDefined2;

template<typename T>
T* reserve(size_t count) {
    void* p = mmap(nullptr, count * sizeof(T), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    return p == MAP_FAILED ? nullptr : static_cast<T*>(p);
}

struct Fault {
    uint32_t kind;
};

}  // namespace

struct sd_machine {
    uint32_t address_bits = 0;
    size_t pages = 0;
    // The page table dynarmic reads directly: the host page of each page that can be read and
    // written. Other pages are null there, so their accesses go to the callbacks, which check
    // the permissions.
    void** fast = nullptr;
    // The host page of every mapped page, and its permissions.
    uint8_t** host = nullptr;
    uint8_t* perm = nullptr;
    std::unique_ptr<Dynarmic::ExclusiveMonitor> monitor;
    std::mutex cores_lock;
    std::vector<sd_core*> cores;

    bool valid_range(uint64_t address, uint64_t size) const {
        if (size == 0 || address % PAGE || size % PAGE) {
            return false;
        }
        const uint64_t end = address + size;
        return end > address && (end - 1) >> address_bits == 0;
    }

    // Checks that every page of [address, address + size) is mapped and allows need.
    std::optional<Fault> check(uint64_t address, uint64_t size, uint8_t need) const {
        const uint64_t last = address + size - 1;
        if (last < address || last >> address_bits) {
            return Fault{SD_UNMAPPED};
        }
        for (uint64_t page = address >> PAGE_BITS; page <= last >> PAGE_BITS; page++) {
            if (!host[page]) {
                return Fault{SD_UNMAPPED};
            }
            if ((perm[page] & need) != need) {
                return Fault{SD_PROTECTED};
            }
        }
        return std::nullopt;
    }

    uint8_t* at(uint64_t address) const {
        return host[address >> PAGE_BITS] + (address & (PAGE - 1));
    }

    // Copies between guest memory and buffer, page by page. The range must have been checked.
    void copy(uint64_t address, void* buffer, uint64_t size, bool to_guest) const {
        auto* b = static_cast<uint8_t*>(buffer);
        while (size) {
            const uint64_t n = std::min(size, PAGE - (address & (PAGE - 1)));
            if (to_guest) {
                std::memcpy(at(address), b, n);
            } else {
                std::memcpy(b, at(address), n);
            }
            address += n;
            b += n;
            size -= n;
        }
    }

    void invalidate(uint64_t address, uint64_t size);
};

struct sd_core final : Dynarmic::A64::UserCallbacks {
    sd_machine* m;
    uint32_t id;
    bool count_ticks;
    uint64_t tpidr = 0;
    uint64_t tpidrro = 0;
    uint64_t counter = 0;
    uint64_t ticks_left = 0;
    uint64_t ticks_run = 0;
    bool stopped = false;
    sd_stop stop{};
    std::unique_ptr<Dynarmic::A64::Jit> jit;

    sd_core(sd_machine* machine, uint32_t processor_id, bool count)
            : m(machine), id(processor_id), count_ticks(count) {
        Dynarmic::A64::UserConfig conf{};
        conf.callbacks = this;
        conf.processor_id = processor_id;
        conf.global_monitor = m->monitor.get();
        conf.page_table = m->fast;
        conf.page_table_address_space_bits = m->address_bits;
        conf.silently_mirror_page_table = false;
        // Accesses that cross a page boundary go to the callbacks, because two guest pages
        // that are next to each other need not be next to each other in host memory.
        conf.detect_misaligned_access_via_page_table = 8 | 16 | 32 | 64 | 128;
        conf.only_detect_misalignment_via_page_table_on_page_boundary = true;
        // A faulting access stops the run at the instruction that made it.
        conf.check_halt_on_memory_access = true;
        conf.tpidr_el0 = &tpidr;
        conf.tpidrro_el0 = &tpidrro;
        conf.cntfrq_el0 = 19'200'000;
        conf.enable_cycle_counting = count_ticks;
        jit = std::make_unique<Dynarmic::A64::Jit>(conf);
    }

    void stop_with(const sd_stop& s, HaltReason hr = STOPPED) {
        if (!stopped) {
            stop = s;
            stopped = true;
        }
        jit->HaltExecution(hr);
    }

    // A data access that faults. MemoryAbort makes the generated code return right after the
    // access, with the PC at the instruction that made it; sd_run reads the PC then.
    void fault(uint64_t address, uint32_t access, Fault f) {
        sd_stop s{};
        s.reason = SD_FAULT;
        s.address = address;
        s.access = access;
        s.kind = f.kind;
        stop_with(s, STOPPED | HaltReason::MemoryAbort);
    }

    template<typename T>
    T read(VAddr address) {
        T value{};
        if (auto f = m->check(address, sizeof(T), SD_READ)) {
            fault(address, SD_ACCESS_READ, *f);
            return value;
        }
        if constexpr (sizeof(T) <= 8) {
            if (address % sizeof(T) == 0) {
                return __atomic_load_n(reinterpret_cast<T*>(m->at(address)), __ATOMIC_RELAXED);
            }
        }
        m->copy(address, &value, sizeof(T), false);
        return value;
    }

    template<typename T>
    void write(VAddr address, T value) {
        if (auto f = m->check(address, sizeof(T), SD_WRITE)) {
            fault(address, SD_ACCESS_WRITE, *f);
            return;
        }
        if constexpr (sizeof(T) <= 8) {
            if (address % sizeof(T) == 0) {
                __atomic_store_n(reinterpret_cast<T*>(m->at(address)), value, __ATOMIC_RELAXED);
                return;
            }
        }
        m->copy(address, &value, sizeof(T), true);
    }

    template<typename T>
    bool write_exclusive(VAddr address, T value, T expected) {
        if (auto f = m->check(address, sizeof(T), SD_WRITE)) {
            fault(address, SD_ACCESS_WRITE, *f);
            return false;
        }
        if (address % sizeof(T)) {
            // Exclusive accesses must be aligned; the architecture makes this an alignment fault.
            fault(address, SD_ACCESS_WRITE, Fault{SD_PROTECTED});
            return false;
        }
        return __atomic_compare_exchange_n(reinterpret_cast<T*>(m->at(address)), &expected, value, false,
                                           __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
    }

    std::optional<std::uint32_t> MemoryReadCode(VAddr address) override {
        if (m->check(address, 4, SD_EXECUTE)) {
            return std::nullopt;
        }
        std::uint32_t word;
        m->copy(address, &word, 4, false);
        return word;
    }

    std::uint8_t MemoryRead8(VAddr a) override { return read<std::uint8_t>(a); }
    std::uint16_t MemoryRead16(VAddr a) override { return read<std::uint16_t>(a); }
    std::uint32_t MemoryRead32(VAddr a) override { return read<std::uint32_t>(a); }
    std::uint64_t MemoryRead64(VAddr a) override { return read<std::uint64_t>(a); }
    Vector MemoryRead128(VAddr a) override { return read<Vector>(a); }

    void MemoryWrite8(VAddr a, std::uint8_t v) override { write(a, v); }
    void MemoryWrite16(VAddr a, std::uint16_t v) override { write(a, v); }
    void MemoryWrite32(VAddr a, std::uint32_t v) override { write(a, v); }
    void MemoryWrite64(VAddr a, std::uint64_t v) override { write(a, v); }
    void MemoryWrite128(VAddr a, Vector v) override { write(a, v); }

    bool MemoryWriteExclusive8(VAddr a, std::uint8_t v, std::uint8_t e) override { return write_exclusive(a, v, e); }
    bool MemoryWriteExclusive16(VAddr a, std::uint16_t v, std::uint16_t e) override { return write_exclusive(a, v, e); }
    bool MemoryWriteExclusive32(VAddr a, std::uint32_t v, std::uint32_t e) override { return write_exclusive(a, v, e); }
    bool MemoryWriteExclusive64(VAddr a, std::uint64_t v, std::uint64_t e) override { return write_exclusive(a, v, e); }
    bool MemoryWriteExclusive128(VAddr a, Vector v, Vector e) override {
        unsigned __int128 value, expected;
        std::memcpy(&value, v.data(), 16);
        std::memcpy(&expected, e.data(), 16);
        return write_exclusive(a, value, expected);
    }

    void InterpreterFallback(VAddr pc, size_t num_instructions) override {
        sd_stop s{};
        s.reason = SD_FALLBACK;
        s.pc = pc;
        s.number = static_cast<uint32_t>(num_instructions);
        stop_with(s);
    }

    void CallSVC(std::uint32_t swi) override {
        // dynarmic has already set the PC past the SVC.
        sd_stop s{};
        s.reason = SD_SVC;
        s.number = swi;
        s.pc = jit->GetPC() - 4;
        stop_with(s);
    }

    void ExceptionRaised(VAddr pc, Exception exception) override {
        sd_stop s{};
        s.pc = pc;
        switch (exception) {
        case Exception::UnallocatedEncoding:
        case Exception::ReservedValue:
        case Exception::UnpredictableInstruction:
            s.reason = SD_UNDEFINED;
            break;
        case Exception::Breakpoint: {
            std::uint32_t word = 0;
            if (!m->check(pc, 4, SD_READ)) {
                m->copy(pc, &word, 4, false);
            }
            s.reason = SD_BREAK;
            s.number = (word >> 5) & 0xFFFF;
            break;
        }
        case Exception::WaitForInterrupt:
        case Exception::WaitForEvent:
        case Exception::SendEvent:
        case Exception::SendEventLocal:
        case Exception::Yield:
            // Hints. dynarmic has already moved the PC past them; they do nothing, as in siicl.cpu.
            return;
        case Exception::NoExecuteFault: {
            const auto f = m->check(pc, 4, SD_EXECUTE);
            s.reason = SD_FAULT;
            s.address = pc;
            s.access = SD_ACCESS_FETCH;
            s.kind = f ? f->kind : SD_PROTECTED;
            break;
        }
        default:
            s.reason = SD_EXCEPTION;
            s.number = static_cast<uint32_t>(exception);
            break;
        }
        stop_with(s);
    }

    void InstructionCacheOperationRaised(Dynarmic::A64::InstructionCacheOperation op, VAddr value) override {
        if (op == Dynarmic::A64::InstructionCacheOperation::InvalidateByVAToPoU) {
            jit->InvalidateCacheRange(value & ~uint64_t{63}, 64);
        } else {
            jit->ClearCache();
        }
    }

    void AddTicks(std::uint64_t ticks) override {
        ticks_run += ticks;
        ticks_left = ticks >= ticks_left ? 0 : ticks_left - ticks;
    }

    std::uint64_t GetTicksRemaining() override { return ticks_left; }

    std::uint64_t GetCNTPCT() override { return counter++; }
};

void sd_machine::invalidate(uint64_t address, uint64_t size) {
    std::lock_guard lock{cores_lock};
    for (sd_core* c : cores) {
        c->jit->InvalidateCacheRange(address, size);
    }
}

extern "C" {

const char* sd_version(void) {
    return "siicl-dynarmic 1, dynarmic 6.7.0";
}

sd_machine* sd_machine_create(uint32_t address_bits, uint32_t cores) {
    if (address_bits < 13 || address_bits > 48 || cores == 0) {
        return nullptr;
    }
    auto m = std::make_unique<sd_machine>();
    m->address_bits = address_bits;
    m->pages = size_t{1} << (address_bits - PAGE_BITS);
    m->fast = reserve<void*>(m->pages);
    m->host = reserve<uint8_t*>(m->pages);
    m->perm = reserve<uint8_t>(m->pages);
    if (!m->fast || !m->host || !m->perm) {
        sd_machine_destroy(m.release());
        return nullptr;
    }
    m->monitor = std::make_unique<Dynarmic::ExclusiveMonitor>(cores);
    return m.release();
}

void sd_machine_destroy(sd_machine* m) {
    if (!m) {
        return;
    }
    if (m->fast) {
        munmap(m->fast, m->pages * sizeof(void*));
    }
    if (m->host) {
        munmap(m->host, m->pages * sizeof(uint8_t*));
    }
    if (m->perm) {
        munmap(m->perm, m->pages);
    }
    delete m;
}

static void set_page(sd_machine* m, uint64_t page, uint8_t* host, uint32_t perm) {
    m->host[page] = host;
    m->perm[page] = host ? static_cast<uint8_t>(perm & 7) : 0;
    const bool direct = host && (perm & (SD_READ | SD_WRITE)) == (SD_READ | SD_WRITE);
    m->fast[page] = direct ? host : nullptr;
}

int sd_map(sd_machine* m, uint64_t address, uint64_t size, uint32_t perm, void* host) {
    if (!m->valid_range(address, size) || !host || reinterpret_cast<uintptr_t>(host) % PAGE) {
        return -1;
    }
    auto* h = static_cast<uint8_t*>(host);
    for (uint64_t i = 0; i < size / PAGE; i++) {
        set_page(m, (address >> PAGE_BITS) + i, h + i * PAGE, perm);
    }
    m->invalidate(address, size);
    return 0;
}

int sd_protect(sd_machine* m, uint64_t address, uint64_t size, uint32_t perm) {
    if (!m->valid_range(address, size) || m->check(address, size, 0)) {
        return -1;
    }
    for (uint64_t page = address >> PAGE_BITS; page < (address + size) >> PAGE_BITS; page++) {
        set_page(m, page, m->host[page], perm);
    }
    m->invalidate(address, size);
    return 0;
}

int sd_unmap(sd_machine* m, uint64_t address, uint64_t size) {
    if (!m->valid_range(address, size)) {
        return -1;
    }
    for (uint64_t page = address >> PAGE_BITS; page < (address + size) >> PAGE_BITS; page++) {
        set_page(m, page, nullptr, 0);
    }
    m->invalidate(address, size);
    return 0;
}

int sd_read(sd_machine* m, uint64_t address, void* out, uint64_t size) {
    if (size == 0) {
        return 0;
    }
    if (m->check(address, size, 0)) {
        return -1;
    }
    m->copy(address, out, size, false);
    return 0;
}

int sd_write(sd_machine* m, uint64_t address, const void* data, uint64_t size) {
    if (size == 0) {
        return 0;
    }
    if (m->check(address, size, 0)) {
        return -1;
    }
    m->copy(address, const_cast<void*>(data), size, true);
    m->invalidate(address, size);
    return 0;
}

sd_core* sd_core_create(sd_machine* m, uint32_t processor_id, int count_ticks) {
    if (processor_id >= m->monitor->GetProcessorCount()) {
        return nullptr;
    }
    std::lock_guard lock{m->cores_lock};
    for (sd_core* c : m->cores) {
        if (c->id == processor_id) {
            return nullptr;
        }
    }
    auto* c = new sd_core(m, processor_id, count_ticks != 0);
    m->cores.push_back(c);
    return c;
}

void sd_core_destroy(sd_core* c) {
    if (!c) {
        return;
    }
    {
        std::lock_guard lock{c->m->cores_lock};
        auto& cores = c->m->cores;
        cores.erase(std::remove(cores.begin(), cores.end(), c), cores.end());
    }
    delete c;
}

uint64_t sd_get_x(sd_core* c, uint32_t i) { return i < 31 ? c->jit->GetRegister(i) : 0; }
void sd_set_x(sd_core* c, uint32_t i, uint64_t v) {
    if (i < 31) {
        c->jit->SetRegister(i, v);
    }
}
uint64_t sd_get_sp(sd_core* c) { return c->jit->GetSP(); }
void sd_set_sp(sd_core* c, uint64_t v) { c->jit->SetSP(v); }
uint64_t sd_get_pc(sd_core* c) { return c->jit->GetPC(); }
void sd_set_pc(sd_core* c, uint64_t v) { c->jit->SetPC(v); }
void sd_get_v(sd_core* c, uint32_t i, uint64_t out[2]) {
    const Vector v = i < 32 ? c->jit->GetVector(i) : Vector{};
    out[0] = v[0];
    out[1] = v[1];
}
void sd_set_v(sd_core* c, uint32_t i, const uint64_t value[2]) {
    if (i < 32) {
        c->jit->SetVector(i, Vector{value[0], value[1]});
    }
}
uint32_t sd_get_pstate(sd_core* c) { return c->jit->GetPstate(); }
void sd_set_pstate(sd_core* c, uint32_t v) { c->jit->SetPstate(v); }
uint32_t sd_get_fpcr(sd_core* c) { return c->jit->GetFpcr(); }
void sd_set_fpcr(sd_core* c, uint32_t v) { c->jit->SetFpcr(v); }
uint32_t sd_get_fpsr(sd_core* c) { return c->jit->GetFpsr(); }
void sd_set_fpsr(sd_core* c, uint32_t v) { c->jit->SetFpsr(v); }
uint64_t sd_get_tpidr_el0(sd_core* c) { return c->tpidr; }
void sd_set_tpidr_el0(sd_core* c, uint64_t v) { c->tpidr = v; }
uint64_t sd_get_tpidrro_el0(sd_core* c) { return c->tpidrro; }
void sd_set_tpidrro_el0(sd_core* c, uint64_t v) { c->tpidrro = v; }
uint64_t sd_get_counter(sd_core* c) { return c->counter; }
void sd_set_counter(sd_core* c, uint64_t v) { c->counter = v; }

void sd_save_context(sd_core* c, sd_context* out) {
    const auto x = c->jit->GetRegisters();
    std::copy(x.begin(), x.end(), out->x);
    out->sp = c->jit->GetSP();
    out->pc = c->jit->GetPC();
    const auto v = c->jit->GetVectors();
    for (size_t i = 0; i < 32; i++) {
        out->v[i][0] = v[i][0];
        out->v[i][1] = v[i][1];
    }
    out->pstate = c->jit->GetPstate();
    out->fpcr = c->jit->GetFpcr();
    out->fpsr = c->jit->GetFpsr();
    out->reserved = 0;
    out->tpidr_el0 = c->tpidr;
    out->tpidrro_el0 = c->tpidrro;
}

void sd_restore_context(sd_core* c, const sd_context* in) {
    std::array<std::uint64_t, 31> x;
    std::copy(in->x, in->x + 31, x.begin());
    c->jit->SetRegisters(x);
    c->jit->SetSP(in->sp);
    c->jit->SetPC(in->pc);
    std::array<Vector, 32> v;
    for (size_t i = 0; i < 32; i++) {
        v[i] = Vector{in->v[i][0], in->v[i][1]};
    }
    c->jit->SetVectors(v);
    c->jit->SetPstate(in->pstate);
    c->jit->SetFpcr(in->fpcr);
    c->jit->SetFpsr(in->fpsr);
    c->tpidr = in->tpidr_el0;
    c->tpidrro = in->tpidrro_el0;
    c->jit->ClearExclusiveState();
}

void sd_run(sd_core* c, uint64_t ticks, sd_stop* out) {
    c->stopped = false;
    c->stop = sd_stop{};
    c->ticks_left = ticks;
    c->ticks_run = 0;
    for (;;) {
        const HaltReason hr = c->jit->Run();
        if (c->stopped) {
            if (c->stop.reason == SD_FAULT && c->stop.access != SD_ACCESS_FETCH) {
                c->stop.pc = c->jit->GetPC();
            } else if (c->stop.reason != SD_SVC) {
                // dynarmic moves the PC past an instruction that raises an exception. Leave it
                // at the instruction, as siicl.cpu does.
                c->jit->SetPC(c->stop.pc);
            }
            break;
        }
        if (Dynarmic::Has(hr, HALTED)) {
            c->stop.reason = SD_HALTED;
            c->stop.pc = c->jit->GetPC();
            break;
        }
        if (c->count_ticks && c->ticks_left == 0) {
            c->stop.reason = SD_LIMIT;
            c->stop.pc = c->jit->GetPC();
            break;
        }
        // Run also returns after a cache invalidation that a callback asked for; go on.
    }
    c->stop.ticks = c->ticks_run;
    *out = c->stop;
}

void sd_halt(sd_core* c) {
    c->jit->HaltExecution(HALTED);
}

void sd_invalidate(sd_core* c, uint64_t address, uint64_t size) {
    c->jit->InvalidateCacheRange(address, size);
}

}  // extern "C"
