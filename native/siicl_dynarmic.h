/* C interface to dynarmic for siicl.
 *
 * siicl loads this library with ctypes (src/siicl/dynarmic.py). A machine holds the guest
 * address space and the exclusive monitor that its cores share; a core is one AArch64 CPU.
 * Guest memory is host memory that the caller owns and maps page by page.
 */
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SD_API __attribute__((visibility("default")))

typedef struct sd_machine sd_machine;
typedef struct sd_core sd_core;

/* Page permissions, with the same values as Unicorn's. */
enum { SD_READ = 1, SD_WRITE = 2, SD_EXECUTE = 4 };

/* Why sd_run returned. */
enum {
    SD_LIMIT = 0,     /* the tick budget ran out; pc is the next instruction */
    SD_SVC = 1,       /* an SVC at pc with number "number"; the core's PC is past it */
    SD_FAULT = 2,     /* an access ("access", "kind") at "address" by the instruction at pc */
    SD_UNDEFINED = 3, /* an instruction dynarmic does not decode, at pc */
    SD_BREAK = 4,     /* a BRK at pc, with its immediate in "number" */
    SD_EXCEPTION = 5, /* another dynarmic exception, with its number in "number" */
    SD_FALLBACK = 6,  /* dynarmic asked for an interpreter for "number" instructions at pc */
    SD_HALTED = 7,    /* sd_halt was called from another thread */
};

enum { SD_ACCESS_READ = 0, SD_ACCESS_WRITE = 1, SD_ACCESS_FETCH = 2 };
enum { SD_UNMAPPED = 0, SD_PROTECTED = 1 };

typedef struct {
    uint32_t reason;
    uint32_t number;
    uint64_t pc;
    uint64_t address;
    uint32_t access;
    uint32_t kind;
    uint64_t ticks; /* ticks the run used */
} sd_stop;

typedef struct {
    uint64_t x[31];
    uint64_t sp;
    uint64_t pc;
    uint64_t v[32][2];
    uint32_t pstate;
    uint32_t fpcr;
    uint32_t fpsr;
    uint32_t reserved;
    uint64_t tpidr_el0;
    uint64_t tpidrro_el0;
} sd_context;

SD_API const char* sd_version(void);

/* address_bits is the size of the guest address space (39 for a game); cores is the most
 * cores the machine will have. Returns NULL if the page tables cannot be reserved. */
SD_API sd_machine* sd_machine_create(uint32_t address_bits, uint32_t cores);
SD_API void sd_machine_destroy(sd_machine* m);

/* Map, change and unmap whole pages. host must stay allocated while it is mapped. Each
 * returns 0, or -1 for an address range that is not page-aligned or not in the address space.
 * These, and sd_write, must not be called while a core of the machine runs. */
SD_API int sd_map(sd_machine* m, uint64_t address, uint64_t size, uint32_t perm, void* host);
SD_API int sd_protect(sd_machine* m, uint64_t address, uint64_t size, uint32_t perm);
SD_API int sd_unmap(sd_machine* m, uint64_t address, uint64_t size);

/* Read and write guest memory whatever its permissions. A write makes every core translate
 * the code in the range again. Each returns 0, or -1 if part of the range is unmapped. */
SD_API int sd_read(sd_machine* m, uint64_t address, void* out, uint64_t size);
SD_API int sd_write(sd_machine* m, uint64_t address, const void* data, uint64_t size);

/* processor_id must be below the machine's core count and unique among its cores.
 *
 * With count_ticks set, sd_run stops after its tick budget, and sd_halt has no effect while
 * the core runs code that loops without leaving its blocks. Without it, dynarmic counts no
 * instructions, sd_run ignores the budget, and a run ends only at a stop or at sd_halt.
 * dynarmic checks one or the other between linked blocks, not both. */
SD_API sd_core* sd_core_create(sd_machine* m, uint32_t processor_id, int count_ticks);
SD_API void sd_core_destroy(sd_core* c);

SD_API uint64_t sd_get_x(sd_core* c, uint32_t index);
SD_API void sd_set_x(sd_core* c, uint32_t index, uint64_t value);
SD_API uint64_t sd_get_sp(sd_core* c);
SD_API void sd_set_sp(sd_core* c, uint64_t value);
SD_API uint64_t sd_get_pc(sd_core* c);
SD_API void sd_set_pc(sd_core* c, uint64_t value);
SD_API void sd_get_v(sd_core* c, uint32_t index, uint64_t out[2]);
SD_API void sd_set_v(sd_core* c, uint32_t index, const uint64_t value[2]);
SD_API uint32_t sd_get_pstate(sd_core* c);
SD_API void sd_set_pstate(sd_core* c, uint32_t value);
SD_API uint32_t sd_get_fpcr(sd_core* c);
SD_API void sd_set_fpcr(sd_core* c, uint32_t value);
SD_API uint32_t sd_get_fpsr(sd_core* c);
SD_API void sd_set_fpsr(sd_core* c, uint32_t value);
SD_API uint64_t sd_get_tpidr_el0(sd_core* c);
SD_API void sd_set_tpidr_el0(sd_core* c, uint64_t value);
SD_API uint64_t sd_get_tpidrro_el0(sd_core* c);
SD_API void sd_set_tpidrro_el0(sd_core* c, uint64_t value);

/* The system counter (CNTPCT_EL0). Each read returns the value and then adds 1 to it. */
SD_API uint64_t sd_get_counter(sd_core* c);
SD_API void sd_set_counter(sd_core* c, uint64_t value);

/* Save and restore all registers, for switching guest threads. Restoring also clears the
 * core's exclusive reservation. */
SD_API void sd_save_context(sd_core* c, sd_context* out);
SD_API void sd_restore_context(sd_core* c, const sd_context* in);

/* Run from the current PC. With count_ticks, run for "ticks" instructions or a little more:
 * dynarmic checks the budget at the end of each block of translated code. ticks must be
 * positive. */
SD_API void sd_run(sd_core* c, uint64_t ticks, sd_stop* out);

/* Make sd_run return SD_HALTED, on a core created without count_ticks. Safe to call from
 * another thread. */
SD_API void sd_halt(sd_core* c);

/* Translate the code in the range again on this core. */
SD_API void sd_invalidate(sd_core* c, uint64_t address, uint64_t size);

#ifdef __cplusplus
}
#endif
