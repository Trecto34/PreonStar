// Time the real inkling2_moe_down_decode_iq4nl.spv in a decode-shaped chain
// (48 dispatches, barrier between) next to a plain streaming read of the same bytes.
// usage: moe moe_down.spv bw.spv [prof]   (prof = profiler-style reset+timestamp pair per dispatch)
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#define CK(x) do { VkResult r_ = (x); if (r_) { fprintf(stderr, #x " = %d\n", r_); exit(1); } } while (0)

enum { OUT = 2304, IN = 2304, MID = 1152, BLOCKS = 9, NUSED = 8, NEXP = 128, LAYERS = 18, ROWB = BLOCKS * 144 };
static VkPhysicalDevice pd; static VkDevice dev; static VkQueue q; static VkCommandBuffer cb; static VkFence fence;
static VkQueryPool qp; static double ns_per_tick;

typedef struct { VkPipeline p; VkPipelineLayout l; VkDescriptorSet s; } kern;

static VkBuffer mkbuf(VkDeviceSize sz) {
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = sz,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    VkBuffer b; CK(vkCreateBuffer(dev, &bi, 0, &b));
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, b, &mr);
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    uint32_t t = 0; while (!((mr.memoryTypeBits >> t) & 1) || !(mp.memoryTypes[t].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) t++;
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size, .memoryTypeIndex = t};
    VkDeviceMemory m; CK(vkAllocateMemory(dev, &ai, 0, &m)); CK(vkBindBufferMemory(dev, b, m, 0));
    return b;
}
static kern mkkern(const char *path, uint32_t nb, VkBuffer *bufs, uint32_t push) {
    FILE *f = fopen(path, "rb"); if (!f) exit(3);
    fseek(f, 0, SEEK_END); long n = ftell(f); rewind(f);
    uint32_t *code = malloc(n); fread(code, 1, n, f); fclose(f);
    VkDescriptorSetLayoutBinding b[8]; VkDescriptorBufferInfo dbi[8];
    for (uint32_t i = 0; i < nb; i++) {
        b[i] = (VkDescriptorSetLayoutBinding){i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT};
        dbi[i] = (VkDescriptorBufferInfo){bufs[i], 0, VK_WHOLE_SIZE};
    }
    kern k; VkDescriptorSetLayout dsl;
    VkDescriptorSetLayoutCreateInfo dl = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = nb, .pBindings = b};
    CK(vkCreateDescriptorSetLayout(dev, &dl, 0, &dsl));
    VkPushConstantRange pr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, push};
    VkPipelineLayoutCreateInfo pl = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1, .pSetLayouts = &dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pr};
    CK(vkCreatePipelineLayout(dev, &pl, 0, &k.l));
    VkShaderModuleCreateInfo sm = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = n, .pCode = code};
    VkShaderModule mod; CK(vkCreateShaderModule(dev, &sm, 0, &mod));
    VkComputePipelineCreateInfo cp = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = mod, .pName = "main"}, .layout = k.l};
    CK(vkCreateComputePipelines(dev, 0, 1, &cp, 0, &k.p));
    VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nb};
    VkDescriptorPoolCreateInfo dpi = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &ps};
    VkDescriptorPool dp; CK(vkCreateDescriptorPool(dev, &dpi, 0, &dp));
    VkDescriptorSetAllocateInfo dai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = dp, .descriptorSetCount = 1, .pSetLayouts = &dsl};
    CK(vkAllocateDescriptorSets(dev, &dai, &k.s));
    VkWriteDescriptorSet w = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = k.s, .descriptorCount = nb, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = dbi};
    vkUpdateDescriptorSets(dev, 1, &w, 0, 0);
    return k;
}
static void begin(void) {
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    CK(vkResetCommandBuffer(cb, 0)); CK(vkBeginCommandBuffer(cb, &bi));
    vkCmdResetQueryPool(cb, qp, 0, 2 + 2 * LAYERS);
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, 0);
}
static void barrier(void) {
    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, 0, 0, 0);
}
static double end_us(double *per_disp) {
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, 1);
    CK(vkEndCommandBuffer(cb));
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb};
    CK(vkResetFences(dev, 1, &fence)); CK(vkQueueSubmit(q, 1, &si, fence));
    CK(vkWaitForFences(dev, 1, &fence, VK_TRUE, ~0ull));
    uint64_t t[2 + 2 * LAYERS];
    uint32_t nq = per_disp ? 2 + 2 * LAYERS : 2;
    CK(vkGetQueryPoolResults(dev, qp, 0, nq, sizeof t, t, 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
    if (per_disp) { double s = 0; for (int i = 0; i < LAYERS; i++) s += (t[3 + 2 * i] - t[2 + 2 * i]) * ns_per_tick / 1000.0; *per_disp = s / LAYERS; }
    return (t[1] - t[0]) * ns_per_tick / 1000.0;
}
static int cmpd(const void *a, const void *b) { double x = *(double *)a, y = *(double *)b; return (x > y) - (x < y); }

int main(int argc, char **argv) {
    int prof = argc > 3;
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

    /* one weight buffer per layer, NEXP experts each: 48 x 191 MB would not fit, so
     * 4 distinct layer buffers reused round-robin (764 MB, >> 4 MB L2). */
    const VkDeviceSize expb = (VkDeviceSize)OUT * ROWB, layerb = expb * NEXP;
    VkBuffer W[4]; for (int i = 0; i < 4; i++) W[i] = mkbuf(layerb);
    VkBuffer act = mkbuf(IN * 4), sel = mkbuf(LAYERS * NUSED * 4 * 64), rw = mkbuf(65536), y = mkbuf(NUSED * MID * 4), es = mkbuf(NEXP * 4);
    kern km[4], ks[4];
    for (int i = 0; i < 4; i++) {
        VkBuffer b6[6] = {W[i], act, sel, y, rw, es}; km[i] = mkkern(argv[1], 6, b6, 32);
        VkBuffer b2[2] = {W[i], y}; ks[i] = mkkern(argv[2], 2, b2, 8);
    }
    begin();
    /* weights: f16 d = 1.0 (0x3C00) pattern so no NaN/denormal paths; quants arbitrary */
    for (int i = 0; i < 4; i++) vkCmdFillBuffer(cb, W[i], 0, VK_WHOLE_SIZE, 0x3C003C00u);
    vkCmdFillBuffer(cb, act, 0, VK_WHOLE_SIZE, 0x3F800000u);
    vkCmdFillBuffer(cb, rw, 0, VK_WHOLE_SIZE, 0x3F800000u);
    vkCmdFillBuffer(cb, es, 0, VK_WHOLE_SIZE, 0x3F800000u);
    end_us(0);
    /* selected experts: distinct pseudo-random per (layer, rep) chunk, uploaded via fill per slot */
    srand(1);

    double tm[3][31], dm[31];
    for (int rep = -300; rep < 31; rep++) {   /* first 10 = clock warm-up */
        /* new expert choice each rep, like new tokens */
        begin();
        for (int s = 0; s < NUSED; s++) vkCmdFillBuffer(cb, sel, s * 4, 4, (uint32_t)((rand() % (NEXP / NUSED)) * NUSED + s));
        end_us(0);
        for (int mode = 0; mode < 3; mode++) {   /* 0 moe_down, 1 stream same expert slices, 2 stream contiguous */
            begin();
            for (int d = 0; d < LAYERS; d++) {
                if (d) barrier();
                if (prof) { vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, qp, 2 + 2 * d); }
                kern *k = mode == 0 ? &km[d & 3] : &ks[d & 3];
                vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, k->p);
                vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, k->l, 0, 1, &k->s, 0, 0);
                if (mode == 0) {
                    struct { uint32_t mid, blocks, n_used; float b, ib, lb, ilb; uint32_t hs; } p8 = {MID, BLOCKS, NUSED, 4.0f, 0.25f, 25.0f, 0.04f, 1};
                    vkCmdPushConstants(cb, k->l, VK_SHADER_STAGE_COMPUTE_BIT, 0, 32, &p8);
                    vkCmdDispatch(cb, MID / 4, NUSED, 1);
                } else {
                    uint32_t nv = (uint32_t)(expb * NUSED / 16);
                    uint32_t base = (uint32_t)(((uint64_t)(rep + 300) * 3 + d) % (NEXP / NUSED - 1) * nv);
                    uint32_t p2[2] = {base, nv};
                    vkCmdPushConstants(cb, k->l, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, p2);
                    vkCmdDispatch(cb, (nv + 1023) / 1024, 1, 1);
                }
                if (prof) vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, qp, 3 + 2 * d);
            }
            double pd_us = 0, t = end_us(prof ? &pd_us : 0);
            if (rep >= 0) { tm[mode][rep] = t; if (mode == 0) dm[rep] = pd_us; }
        }
    }
    const char *name[3] = {"gate_up kernel", "stream, same bytes, 1 block", "-"};
    double mb = expb * NUSED / 1e6;
    for (int mode = 0; mode < 2; mode++) {
        qsort(tm[mode], 31, sizeof(double), cmpd);
        double m = tm[mode][15];
        printf("%-30s %8.1f us/48  %6.1f us/disp  %6.1f GB/s  (%.2f MB/disp)\n", name[mode], m, m / LAYERS, LAYERS * mb / m * 1e3, mb);
    }
    if (prof) { qsort(dm, 31, sizeof(double), cmpd); printf("moe_down per-dispatch timestamp bracket: %.1f us\n", dm[15]); }
    return 0;
}
