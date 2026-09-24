/* Interleaved multi-variant probe for the PQ2_0 repack question.
 *
 * One process, one large buffer, N pipelines from the archived mv3.comp
 * variants.  Reps are interleaved variant-major so drift lands on every
 * variant equally; medians are reported, not means.
 *
 * Each dispatch covers `window_rows` rows (=> window_rows/4 workgroups of 64
 * lanes, 4 rows each: the production decode geometry) and the window walks a
 * DRAM-sized buffer over `inner = total_rows/window_rows` dispatches.  The
 * window keeps the per-dispatch shape of a real decode dispatch while the
 * aggregate footprint defeats L2 residency, which a small buffer does not.
 *
 * usage: probe_mv3 <total_rows> <window_rows> <q8k_blocks_per_row> <spv>...
 *   bytes are charged at window_rows x q8k_blocks x 2 x 34 (the production
 *   PQ2_0 stride) for every variant, SoA included. */
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { fprintf(stderr, "%s -> %d\n", #x, r_); exit(1); } } while (0)
#define MAXV 8

static VkDevice dev;
static VkPhysicalDevice pd;

static unsigned char *slurp(const char *p, size_t *n) {
    FILE *f = fopen(p, "rb"); if (!f) { perror(p); exit(1); }
    fseek(f, 0, SEEK_END); *n = ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char *b = malloc(*n); fread(b, 1, *n, f); fclose(f); return b;
}

static int cmpd(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y ? 1 : 0;
}

int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: probe_mv3 <total_rows> <window_rows> <blocks> <spv>...\n"); return 2; }
    uint32_t total_rows = (uint32_t)strtoul(argv[1], NULL, 10);
    uint32_t window_rows = (uint32_t)strtoul(argv[2], NULL, 10);
    uint32_t blocks = (uint32_t)strtoul(argv[3], NULL, 10);
    const int nv = argc - 4;
    if (nv > MAXV) { fprintf(stderr, "too many variants\n"); return 2; }
    if (!window_rows || total_rows % window_rows) { fprintf(stderr, "total_rows must be a multiple of window_rows\n"); return 2; }
    int reps;
    { const char *re = getenv("REPS"); int r_ = re ? atoi(re) : 10; if (r_ < 1) r_ = 1; if (r_ > 16) r_ = 16; reps = r_; }
    const uint32_t row_bytes = blocks * 2u * 34u;
    const uint32_t inner = total_rows / window_rows;
    const uint32_t groups = window_rows / 4u;
    const VkDeviceSize bytes = (VkDeviceSize)total_rows * row_bytes;

    VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &(VkApplicationInfo){ .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_1 } };
    VkInstance inst; CK(vkCreateInstance(&ici, NULL, &inst));
    uint32_t n = 1; CK(vkEnumeratePhysicalDevices(inst, &n, &pd));
    VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(pd, &props);
    float ts_period = props.limits.timestampPeriod;
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = 0, .queueCount = 1, .pQueuePriorities = &prio };
    VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci };
    CK(vkCreateDevice(pd, &dci, NULL, &dev));
    VkQueue q; vkGetDeviceQueue(dev, 0, 0, &q);

    VkDescriptorSetLayoutBinding b[3] = {
        { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
        { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
        { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL } };
    VkDescriptorSetLayout dsl; CK(vkCreateDescriptorSetLayout(dev, &(VkDescriptorSetLayoutCreateInfo){ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 3, .pBindings = b }, NULL, &dsl));
    VkPushConstantRange pcr = { VK_SHADER_STAGE_COMPUTE_BIT, 0, 12 };
    VkPipelineLayout pl; CK(vkCreatePipelineLayout(dev, &(VkPipelineLayoutCreateInfo){ .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1, .pSetLayouts = &dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr }, NULL, &pl));

    VkBuffer buf; CK(vkCreateBuffer(dev, &(VkBufferCreateInfo){ .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = bytes, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT }, NULL, &buf));
    VkMemoryRequirements req; vkGetBufferMemoryRequirements(dev, buf, &req);
    if (!(req.memoryTypeBits & (1u << 3))) { fprintf(stderr, "type 3 unsupported\n"); return 1; }
    VkDeviceMemory mem; CK(vkAllocateMemory(dev, &(VkMemoryAllocateInfo){ .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = req.size, .memoryTypeIndex = 3 }, NULL, &mem));
    CK(vkBindBufferMemory(dev, buf, mem, 0));

    const VkDeviceSize abytes = 1u << 20;
    VkBuffer ab; CK(vkCreateBuffer(dev, &(VkBufferCreateInfo){ .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = abytes, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT }, NULL, &ab));
    VkMemoryRequirements areq; vkGetBufferMemoryRequirements(dev, ab, &areq);
    VkDeviceMemory am; CK(vkAllocateMemory(dev, &(VkMemoryAllocateInfo){ .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = areq.size, .memoryTypeIndex = 3 }, NULL, &am));
    CK(vkBindBufferMemory(dev, ab, am, 0));

    VkBuffer ob; CK(vkCreateBuffer(dev, &(VkBufferCreateInfo){ .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = 65536, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT }, NULL, &ob));
    VkMemoryRequirements oreq; vkGetBufferMemoryRequirements(dev, ob, &oreq);
    VkDeviceMemory om; CK(vkAllocateMemory(dev, &(VkMemoryAllocateInfo){ .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = oreq.size, .memoryTypeIndex = __builtin_ctz(oreq.memoryTypeBits) }, NULL, &om));
    CK(vkBindBufferMemory(dev, ob, om, 0));

    VkDescriptorPool dp; CK(vkCreateDescriptorPool(dev, &(VkDescriptorPoolCreateInfo){ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = MAXV, .poolSizeCount = 1,
        .pPoolSizes = &(VkDescriptorPoolSize){ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3u * MAXV } }, NULL, &dp));
    VkCommandPool cp; CK(vkCreateCommandPool(dev, &(VkCommandPoolCreateInfo){ .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = 0 }, NULL, &cp));
    VkQueryPool qp; CK(vkCreateQueryPool(dev, &(VkQueryPoolCreateInfo){ .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = 2 }, NULL, &qp));
    /* Production reports gpu_ms as the SUM of exclusive per-dispatch timestamp
     * pairs (q36_vulkan.c:3185-3190), so inter-dispatch bubbles are excluded.
     * One pair around the whole loop includes them.  Measure both. */
    if (2u * inner + 2u > 8000u) { fprintf(stderr, "inner too large for pair timestamps\n"); return 2; }
    VkQueryPool qp2; CK(vkCreateQueryPool(dev, &(VkQueryPoolCreateInfo){ .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = 2u * inner + 2u }, NULL, &qp2));

    VkPipeline pipe[MAXV]; VkDescriptorSet ds[MAXV];
    for (int v = 0; v < nv; v++) {
        size_t spv_n; unsigned char *spv = slurp(argv[4 + v], &spv_n);
        VkShaderModule sm; CK(vkCreateShaderModule(dev, &(VkShaderModuleCreateInfo){ .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = spv_n, .pCode = (uint32_t *)spv }, NULL, &sm));
        CK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &(VkComputePipelineCreateInfo){ .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = sm, .pName = "main" }, .layout = pl }, NULL, &pipe[v]));
        CK(vkAllocateDescriptorSets(dev, &(VkDescriptorSetAllocateInfo){ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = dp, .descriptorSetCount = 1, .pSetLayouts = &dsl }, &ds[v]));
        VkDescriptorBufferInfo bi[3] = { { buf, 0, VK_WHOLE_SIZE }, { ab, 0, VK_WHOLE_SIZE }, { ob, 0, VK_WHOLE_SIZE } };
        VkWriteDescriptorSet w[3];
        for (int j = 0; j < 3; j++)
            w[j] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds[v], .dstBinding = (uint32_t)j,
                                           .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bi[j] };
        vkUpdateDescriptorSets(dev, 3, w, 0, NULL);
    }

    VkCommandBuffer cb; CK(vkAllocateCommandBuffers(dev, &(VkCommandBufferAllocateInfo){ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = cp, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1 }, &cb));
    CK(vkBeginCommandBuffer(cb, &(VkCommandBufferBeginInfo){ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO }));
    vkCmdFillBuffer(cb, buf, 0, VK_WHOLE_SIZE, 0x01020304u);
    vkCmdFillBuffer(cb, ab, 0, VK_WHOLE_SIZE, 0x05060708u);
    CK(vkEndCommandBuffer(cb));
    CK(vkQueueSubmit(q, 1, &(VkSubmitInfo){ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb }, VK_NULL_HANDLE));
    CK(vkQueueWaitIdle(q));

    double t[MAXV][16], tp[MAXV][16];
    for (int v = 0; v < nv; v++) for (int r = 0; r < reps; r++) t[v][r] = tp[v][r] = 0;

    /* warm-up pass per variant */
    for (int v = 0; v < nv; v++) {
        CK(vkResetCommandBuffer(cb, 0));
        CK(vkBeginCommandBuffer(cb, &(VkCommandBufferBeginInfo){ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO }));
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe[v]);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds[v], 0, NULL);
        for (uint32_t i = 0; i < inner; i++) {
            uint32_t push[3] = { window_rows, blocks, i * window_rows };
            vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 12, push);
            vkCmdDispatch(cb, groups, 1, 1);
        }
        CK(vkEndCommandBuffer(cb));
        CK(vkQueueSubmit(q, 1, &(VkSubmitInfo){ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb }, VK_NULL_HANDLE));
        CK(vkQueueWaitIdle(q));
    }

    for (int r = 0; r < reps; r++) {
        for (int v = 0; v < nv; v++) {
            CK(vkResetCommandBuffer(cb, 0));
            CK(vkBeginCommandBuffer(cb, &(VkCommandBufferBeginInfo){ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO }));
            vkCmdResetQueryPool(cb, qp, 0, 2);
            vkCmdResetQueryPool(cb, qp2, 0, 2u * inner + 2u);
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe[v]);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds[v], 0, NULL);
            vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, 0);
            for (uint32_t i = 0; i < inner; i++) {
                uint32_t push[3] = { window_rows, blocks, i * window_rows };
                vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, qp2, 2u * i);
                vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 12, push);
                vkCmdDispatch(cb, groups, 1, 1);
                vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp2, 2u * i + 1u);
            }
            vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, 1);
            CK(vkEndCommandBuffer(cb));
            CK(vkQueueSubmit(q, 1, &(VkSubmitInfo){ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb }, VK_NULL_HANDLE));
            CK(vkQueueWaitIdle(q));
            uint64_t ts[2], ts2[8002];
            CK(vkGetQueryPoolResults(dev, qp, 0, 2, sizeof ts, ts, 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
            CK(vkGetQueryPoolResults(dev, qp2, 0, 2u * inner, sizeof ts2, ts2, 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
            t[v][r] = (double)(ts[1] - ts[0]) * ts_period * 1e-6;
            double pair = 0;
            for (uint32_t i = 0; i < inner; i++) pair += (double)(ts2[2u * i + 1u] - ts2[2u * i]);
            tp[v][r] = pair * ts_period * 1e-6;
        }
    }

    printf("total_rows %u  window_rows %u (%u groups/dispatch x %u dispatches)  blocks %u  row_bytes %u  buffer %.2f GiB\n",
           total_rows, window_rows, groups, inner, blocks, row_bytes, (double)bytes / 1073741824.0);
    const double byt = (double)total_rows * row_bytes;
    for (int v = 0; v < nv; v++) {
        double s[16], s2[16]; memcpy(s, t[v], sizeof(double) * reps); memcpy(s2, tp[v], sizeof(double) * reps);
        qsort(s, reps, sizeof(double), cmpd); qsort(s2, reps, sizeof(double), cmpd);
        double med = s[reps / 2], best = s[0];
        double gbs = byt / (med * 1e-3) / 1e9;
        double gbs_best = byt / (best * 1e-3) / 1e9;
        /* pair = production's exclusive per-dispatch convention */
        double gbs_pair = byt / (s2[reps / 2] * 1e-3) / 1e9;
        double gbs_pair_best = byt / (s2[0] * 1e-3) / 1e9;
        printf("  %-20s span med %6.1f GB/s best %6.1f | pairs med %6.1f GB/s best %6.1f | spread %4.1f%%  %.4f ms/dispatch\n",
               argv[4 + v], gbs, gbs_best, gbs_pair, gbs_pair_best, 100.0 * (s[reps - 1] - s[0]) / med, s2[reps / 2] / inner);
    }
    return 0;
}
