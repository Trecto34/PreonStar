// Validate the split/step-major IQ4_NL MoE-down layout on real MiniPlus-R2 weights:
//  1. bit-identity: old kernel + GGUF layout vs new kernel + repacked layout
//  2. sanity: old kernel vs a double-precision CPU reference
//  3. timing: 48-dispatch barrier chain on the real weights, old vs new, interleaved
// usage: validate model.gguf old.spv new.spv off0 off1 off2 off3
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#define CK(x) do { VkResult r_ = (x); if (r_) { fprintf(stderr, #x " = %d\n", r_); exit(1); } } while (0)

enum { OUT = 2304, IN = 1152, BLOCKS = 36, NUSED = 8, NEXP = 128, LAYERS = 48, ROWB = BLOCKS * 18, NL = 4 };
static VkPhysicalDevice pd; static VkDevice dev; static VkQueue q; static VkCommandBuffer cb; static VkFence fence;
static VkQueryPool qp; static double ns_per_tick;
typedef struct { VkPipeline p; VkPipelineLayout l; VkDescriptorSet s; } kern;
#include "common.inc"

static const float LUT[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
static float h2f(uint16_t h) {
    uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
    float v = e == 0 ? ldexpf((float)m, -24) : e == 31 ? INFINITY : ldexpf((float)(m | 1024), (int)e - 25);
    return s ? -v : v;
}
static struct { VkBuffer b; VkDeviceMemory m; void *p; } hostbuf(VkDeviceSize sz, VkBufferUsageFlags use) {
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = sz, .usage = use};
    VkBuffer b; CK(vkCreateBuffer(dev, &bi, 0, &b));
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, b, &mr);
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    uint32_t t = 0; while (!((mr.memoryTypeBits >> t) & 1) || (mp.memoryTypes[t].propertyFlags & want) != want) t++;
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size, .memoryTypeIndex = t};
    VkDeviceMemory m; CK(vkAllocateMemory(dev, &ai, 0, &m)); CK(vkBindBufferMemory(dev, b, m, 0));
    void *p; CK(vkMapMemory(dev, m, 0, VK_WHOLE_SIZE, 0, &p));
    return (typeof(hostbuf(0, 0))){b, m, p};
}
static void upload(VkBuffer dst, const void *src, VkDeviceSize sz) {
    typeof(hostbuf(0, 0)) st = hostbuf(sz, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    memcpy(st.p, src, sz);
    begin(); VkBufferCopy c = {0, 0, sz}; vkCmdCopyBuffer(cb, st.b, dst, 1, &c); end_us(0);
    vkDestroyBuffer(dev, st.b, 0); vkFreeMemory(dev, st.m, 0);
}
static void repack(const uint8_t *src, uint8_t *dst) {
    const size_t eb = (size_t)OUT * ROWB, qbytes = (size_t)OUT * BLOCKS * 16;
    for (size_t e = 0; e < NEXP; e++)
        for (size_t r = 0; r < OUT; r++)
            for (size_t b = 0; b < BLOCKS; b++) {
                const uint8_t *s = src + e * eb + (r * BLOCKS + b) * 18;
                size_t lin = ((b >> 2) * OUT + r) * 4 + (b & 3);
                memcpy(dst + e * eb + lin * 16, s + 2, 16);
                memcpy(dst + e * eb + qbytes + lin * 2, s, 2);
            }
}
static void run_once(kern *k, int layer, int nd, int bar) {
    for (int d = 0; d < nd; d++) {
        if (bar && d) barrier();
        kern *kk = &k[(layer + d) % NL];
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, kk->p);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, kk->l, 0, 1, &kk->s, 0, 0);
        uint32_t p5[5] = {OUT, IN, BLOCKS, NUSED, 1};
        vkCmdPushConstants(cb, kk->l, VK_SHADER_STAGE_COMPUTE_BIT, 0, 20, p5);
        vkCmdDispatch(cb, OUT / 16, NUSED, 1);
    }
}

int main(int argc, char **argv) {
    if (argc < 8) return 2;
    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_2};
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app};
    VkInstance inst; CK(vkCreateInstance(&ici, 0, &inst));
    uint32_t pc = 1; vkEnumeratePhysicalDevices(inst, &pc, &pd);
    VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(pd, &props); ns_per_tick = props.limits.timestampPeriod;
    float prio = 1;
    VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueCount = 1, .pQueuePriorities = &prio};
    VkPhysicalDeviceVulkan11Features f11 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, .storageBuffer16BitAccess = 1};
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &f11, .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci};
    CK(vkCreateDevice(pd, &dci, 0, &dev)); vkGetDeviceQueue(dev, 0, 0, &q);
    VkCommandPoolCreateInfo cpi = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT};
    VkCommandPool pool; CK(vkCreateCommandPool(dev, &cpi, 0, &pool));
    VkCommandBufferAllocateInfo cai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = pool, .commandBufferCount = 1};
    CK(vkAllocateCommandBuffers(dev, &cai, &cb));
    VkFenceCreateInfo fci = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; CK(vkCreateFence(dev, &fci, 0, &fence));
    VkQueryPoolCreateInfo qpi = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = 2 + 2 * LAYERS};
    CK(vkCreateQueryPool(dev, &qpi, 0, &qp));

    const size_t lb = (size_t)NEXP * OUT * ROWB;
    int fd = open(argv[1], O_RDONLY); if (fd < 0) return 3;
    uint8_t *raw = malloc(lb), *rp = malloc(lb);
    uint8_t *keep = malloc(lb);   /* layer 0 original bytes, for the CPU reference */
    VkBuffer Wo[NL], Wn[NL];
    typeof(hostbuf(0, 0)) act = hostbuf(NUSED * IN * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT),
        sel = hostbuf(NUSED * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT), rw = hostbuf(NUSED * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT),
        y = hostbuf(NUSED * OUT * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT), es = hostbuf(NEXP * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    for (int l = 0; l < NL; l++) {
        off_t off = strtoll(argv[4 + l], 0, 10);
        if (pread(fd, raw, lb, off) != (ssize_t)lb) { fprintf(stderr, "short read\n"); return 4; }
        if (l == 0) memcpy(keep, raw, lb);
        repack(raw, rp);
        Wo[l] = mkbuf(lb); Wn[l] = mkbuf(lb);
        upload(Wo[l], raw, lb); upload(Wn[l], rp, lb);
    }
    free(raw); free(rp); close(fd);
    kern ko[NL], kn[NL];
    for (int l = 0; l < NL; l++) {
        VkBuffer bo[6] = {Wo[l], act.b, sel.b, rw.b, y.b, es.b}; ko[l] = mkkern(argv[2], 6, bo, 20);
        VkBuffer bn[6] = {Wn[l], act.b, sel.b, rw.b, y.b, es.b}; kn[l] = mkkern(argv[3], 6, bn, 20);
    }

    /* ---- 1+2: correctness, 64 random cases spread over the 4 layers ---- */
    srand(12345);
    float *yo = malloc(NUSED * OUT * 4), *yn = malloc(NUSED * OUT * 4);
    int ident = 0; double worst_rel = 0; long nz = 0;
    for (int c = 0; c < 64; c++) {
        int layer = c % NL;
        float *a = act.p; for (int i = 0; i < NUSED * IN; i++) a[i] = ((rand() / (float)RAND_MAX) - 0.5f) * 4.0f;
        uint32_t *s = sel.p;
        for (int i = 0; i < NUSED; i++) { int dup; do { s[i] = rand() % NEXP; dup = 0; for (int j = 0; j < i; j++) dup |= s[j] == s[i]; } while (dup); }
        if (c == 0) s[0] = 0, s[7] = NEXP - 1;   /* first and last expert: edge of the buffer */
        float *r = rw.p; for (int i = 0; i < NUSED; i++) r[i] = rand() / (float)RAND_MAX;
        float *e = es.p; for (int i = 0; i < NEXP; i++) e[i] = 0.5f + rand() / (float)RAND_MAX;
        memset(y.p, 0, NUSED * OUT * 4); begin(); run_once(ko, layer, 1, 0); end_us(0); memcpy(yo, y.p, NUSED * OUT * 4);
        memset(y.p, 0, NUSED * OUT * 4); begin(); run_once(kn, layer, 1, 0); end_us(0); memcpy(yn, y.p, NUSED * OUT * 4);
        if (!memcmp(yo, yn, NUSED * OUT * 4)) ident++;
        else { for (int i = 0; i < NUSED * OUT; i++) if (memcmp(&yo[i], &yn[i], 4)) { fprintf(stderr, "case %d first diff at %d: %.9g vs %.9g\n", c, i, yo[i], yn[i]); break; } }
        for (int i = 0; i < NUSED * OUT; i++) nz += yo[i] != 0.0f && isfinite(yo[i]);
        if (layer == 0) {   /* CPU reference for the old kernel on GGUF bytes */
            for (int sl = 0; sl < NUSED; sl++) for (int row = 0; row < OUT; row++) {
                const uint8_t *rowp = keep + ((size_t)s[sl] * OUT + row) * ROWB; double acc = 0, mag = 0;
                for (int b = 0; b < BLOCKS; b++) {
                    const uint8_t *blk = rowp + b * 18; double d = h2f(blk[0] | blk[1] << 8);
                    for (int j = 0; j < 16; j++) {
                        double x0 = a[sl * IN + b * 32 + j], x1 = a[sl * IN + b * 32 + 16 + j];
                        acc += d * (LUT[blk[2 + j] & 15] * x0 + LUT[blk[2 + j] >> 4] * x1);
                        mag += fabs(d) * (fabs(LUT[blk[2 + j] & 15] * x0) + fabs(LUT[blk[2 + j] >> 4] * x1));
                    }
                }
                double ref = acc * r[sl] * e[s[sl]];
                double rel = fabs(yo[sl * OUT + row] - ref) / (mag * fabs(r[sl] * e[s[sl]]) + 1e-30);
                if (rel > worst_rel) worst_rel = rel;
            }
        }
    }
    printf("bit-identical cases: %d/64   nonzero finite outputs: %ld/%d   old-vs-CPU worst error (rel. to sum|terms|): %.2e\n",
           ident, nz, 64 * NUSED * OUT, worst_rel);

    /* ---- 3: timing on real weights, decode-shaped chain, interleaved old/new ---- */
    double t[2][41];
    for (int rep = -300; rep < 41; rep++) {
        uint32_t *s = sel.p; for (int i = 0; i < NUSED; i++) s[i] = (rand() % (NEXP / NUSED)) * NUSED + i;
        for (int m = 0; m < 2; m++) {
            begin(); run_once(m ? kn : ko, rep & 3, LAYERS, 1); double us = end_us(0);
            if (rep >= 0) t[m][rep] = us / LAYERS;
        }
    }
    for (int m = 0; m < 2; m++) qsort(t[m], 41, sizeof(double), cmpd);
    double mb = (double)NUSED * OUT * ROWB / 1e6;
    printf("old layout+kernel: median %.2f us/disp (p10 %.2f p90 %.2f)  %.1f GB/s\n", t[0][20], t[0][4], t[0][36], mb / t[0][20] * 1e3);
    printf("new layout+kernel: median %.2f us/disp (p10 %.2f p90 %.2f)  %.1f GB/s   delta %+.1f%%\n", t[1][20], t[1][4], t[1][36], mb / t[1][20] * 1e3,
           100.0 * (t[1][20] / t[0][20] - 1));
    return 0;
}
