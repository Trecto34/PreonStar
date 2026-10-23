#ifndef P36_IQ2_TABLES_VULKAN_INC
#define P36_IQ2_TABLES_VULKAN_INC

#include <string.h>

#include "p36_iq_tables.h"

/* Vulkan table payload for the Qwen3.6 AntirezExperts GGUF.  The shipped
 * model needs IQ2_XXS for routed gate/up experts; the same shader buffer also
 * carries IQ2_S/IQ3_S codebooks for the generic routed matvec path. */
enum {
    P36_VK_TAB_XXS_BYTES = 2048,    /* 256 grid entries x 8 bytes */
    P36_VK_TAB_KSIGNS_BYTES = 128,
    P36_VK_TAB_IQ2S_BYTES = 8192,   /* 1024 entries x 8 bytes */
    P36_VK_TAB_IQ3S_BYTES = 2048,   /* 512 entries x 4 bytes */
    P36_VK_TAB_IQ3XXS_BYTES = 1024, /* 256 entries x 4 bytes */
    P36_VK_TAB_IQ1S_BYTES = 16384,  /* 2048 entries x 8 bytes */
    P36_VK_TAB_IQ2XS_BYTES = 4096,  /* 512 entries x 8 bytes */
    P36_VK_TAB_TOTAL_BYTES = P36_VK_TAB_XXS_BYTES +
                             P36_VK_TAB_KSIGNS_BYTES +
                             P36_VK_TAB_IQ2S_BYTES +
                             P36_VK_TAB_IQ3S_BYTES +
                             P36_VK_TAB_IQ3XXS_BYTES +
                             P36_VK_TAB_IQ1S_BYTES +
                             P36_VK_TAB_IQ2XS_BYTES,
};

/* p36_iq2xxs_grid is a string literal, so sizeof carries one trailing NUL. */
typedef char p36_vk_iq_table_sizes[
    (sizeof(p36_iq2xxs_grid) == P36_VK_TAB_XXS_BYTES + 1 &&
     sizeof(p36_ksigns_iq2xs) == P36_VK_TAB_KSIGNS_BYTES &&
     sizeof(p36_iq2s_grid) == P36_VK_TAB_IQ2S_BYTES &&
     sizeof(p36_iq3s_grid) == P36_VK_TAB_IQ3S_BYTES &&
     sizeof(p36_iq3xxs_grid) == P36_VK_TAB_IQ3XXS_BYTES &&
     sizeof(p36_iq2xs_grid) == P36_VK_TAB_IQ2XS_BYTES &&
     sizeof(p36_kmask_iq2xs) == 8) ? 1 : -1];

/* Layout must match TAB_* offsets in vulkan/moe_*.comp. */
static void p36_vk_pack_iq_tables(unsigned char *p) {
    memcpy(p, p36_iq2xxs_grid, P36_VK_TAB_XXS_BYTES);
    p += P36_VK_TAB_XXS_BYTES;
    memcpy(p, p36_ksigns_iq2xs, P36_VK_TAB_KSIGNS_BYTES);
    p += P36_VK_TAB_KSIGNS_BYTES;
    memcpy(p, p36_iq2s_grid, P36_VK_TAB_IQ2S_BYTES);
    p += P36_VK_TAB_IQ2S_BYTES;
    memcpy(p, p36_iq3s_grid, P36_VK_TAB_IQ3S_BYTES);
    p += P36_VK_TAB_IQ3S_BYTES;
    memcpy(p, p36_iq3xxs_grid, P36_VK_TAB_IQ3XXS_BYTES);
    p += P36_VK_TAB_IQ3XXS_BYTES;
    p36_quant_pack_iq1_grid(p);
    p += P36_VK_TAB_IQ1S_BYTES;
    memcpy(p, p36_iq2xs_grid, P36_VK_TAB_IQ2XS_BYTES);
}

#endif
