// Read-bandwidth vs dispatch size on RADV. usage: bw bw.spv
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CK(x) do { VkResult r_ = (x); if (r_) { fprintf(stderr, #x " = %d\n", r_); exit(1); } } while (0)
#define BUF_MB 512u

static VkDevice dev; static VkQueue q; static VkCommandBuffer cb; static VkFence fence;
static VkPipeline pipe; static VkPipelineLayout layout; static VkDescriptorSet ds;
static VkQueryPool qp; static double ns_per_tick;

static uint32_t memtype(VkPhysicalDevice pd, uint32_t bits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
    exit(5);
}
static VkBuffer mkbuf(VkPhysicalDevice pd, VkDeviceSize sz) {
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = sz,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    VkBuffer b; CK(vkCreateBuffer(dev, &bi, 0, &b));
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, b, &mr);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size,
        .memoryTypeIndex = memtype(pd, mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
    VkDeviceMemory m; CK(vkAllocateMemory(dev, &ai, 0, &m)); CK(vkBindBufferMemory(dev, b, m, 0));
    return b;
}
static void begin(void) {
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    CK(vkResetCommandBuffer(cb, 0)); CK(vkBeginCommandBuffer(cb, &bi));
    vkCmdResetQueryPool(cb, qp, 0, 2);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &ds, 0, 0);
}
static void barrier(void) {
    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, 0, 0, 0);
}
static void disp(uint32_t base, uint32_t n, uint32_t wgs) {
    uint32_t pc[2] = {base, n};
    vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, pc);
    vkCmdDispatch(cb, wgs, 1, 1);
}
static double end_us(void) {
    CK(vkEndCommandBuffer(cb));
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb};
    CK(vkResetFences(dev, 1, &fence)); CK(vkQueueSubmit(q, 1, &si, fence));
    CK(vkWaitForFences(dev, 1, &fence, VK_TRUE, ~0ull));
    uint64_t t[2]; CK(vkGetQueryPoolResults(dev, qp, 0, 2, sizeof t, t, 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
    return (t[1] - t[0]) * ns_per_tick / 1000.0;
}
static int cmpd(const void *a, const void *b) { double x = *(double *)a, y = *(double *)b; return (x > y) - (x < y); }
static double median(double *v, int n) { qsort(v, n, sizeof *v, cmpd); return v[n / 2]; }

int main(int argc, char **argv) {
    FILE *f = fopen(argv[1], "rb"); fseek(f, 0, SEEK_END); long n = ftell(f); rewind(f);
    uint32_t *code = malloc(n); fread(code, 1, n, f); fclose(f);
    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_2};
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app};
    VkInstance inst; CK(vkCreateInstance(&ici, 0, &inst));
    uint32_t pc = 1; VkPhysicalDevice pd; vkEnumeratePhysicalDevices(inst, &pc, &pd);
    VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(pd, &props);
    ns_per_tick = props.limits.timestampPeriod;
    float prio = 1;
    VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueCount = 1, .pQueuePriorities = &prio};
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci};
    CK(vkCreateDevice(pd, &dci, 0, &dev)); vkGetDeviceQueue(dev, 0, 0, &q);

    VkBuffer A = mkbuf(pd, (VkDeviceSize)BUF_MB << 20), O = mkbuf(pd, 4096);
    VkDescriptorSetLayoutBinding b[2] = {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
                                         {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT}};
    VkDescriptorSetLayoutCreateInfo dl = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 2, .pBindings = b};
    VkDescriptorSetLayout dsl; CK(vkCreateDescriptorSetLayout(dev, &dl, 0, &dsl));
    VkPushConstantRange pr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 8};
    VkPipelineLayoutCreateInfo pl = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1, .pSetLayouts = &dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pr};
    CK(vkCreatePipelineLayout(dev, &pl, 0, &layout));
    VkShaderModuleCreateInfo sm = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = n, .pCode = code};
    VkShaderModule mod; CK(vkCreateShaderModule(dev, &sm, 0, &mod));
    VkComputePipelineCreateInfo cp = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = mod, .pName = "main"}, .layout = layout};
    CK(vkCreateComputePipelines(dev, 0, 1, &cp, 0, &pipe));
    VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};
    VkDescriptorPoolCreateInfo dpi = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &ps};
    VkDescriptorPool dp; CK(vkCreateDescriptorPool(dev, &dpi, 0, &dp));
    VkDescriptorSetAllocateInfo dai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = dp, .descriptorSetCount = 1, .pSetLayouts = &dsl};
    CK(vkAllocateDescriptorSets(dev, &dai, &ds));
    VkDescriptorBufferInfo dbi[2] = {{A, 0, VK_WHOLE_SIZE}, {O, 0, VK_WHOLE_SIZE}};
    VkWriteDescriptorSet w = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .descriptorCount = 2, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = dbi};
    vkUpdateDescriptorSets(dev, 1, &w, 0, 0);
    VkCommandPoolCreateInfo cpi = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT};
    VkCommandPool pool; CK(vkCreateCommandPool(dev, &cpi, 0, &pool));
    VkCommandBufferAllocateInfo cai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = pool, .commandBufferCount = 1};
    CK(vkAllocateCommandBuffers(dev, &cai, &cb));
    VkFenceCreateInfo fci = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; CK(vkCreateFence(dev, &fci, 0, &fence));
    VkQueryPoolCreateInfo qpi = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = 2};
    CK(vkCreateQueryPool(dev, &qpi, 0, &qp));

    const uint32_t total_v4 = (BUF_MB << 20) / 16;
    /* fill so pages are really backed */
    begin(); vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, 0);
    vkCmdFillBuffer(cb, A, 0, VK_WHOLE_SIZE, 0x01020304u);
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, 1); end_us();
    /* warm the clock governor: ~1 s of streaming */
    for (int i = 0; i < 60; i++) { begin(); vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, 0);
        disp(0, total_v4, 40 * 32); vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, 1); end_us(); }

    printf("single dispatch, read-only, median of 41 (offset rotates through %u MiB)\n", BUF_MB);
    printf("%8s %10s %10s %10s %10s\n", "MB", "k=1 GB/s", "k=4", "k=16", "us(k=4)");
    const double mbs[] = {0.5, 2, 6, 12, 24, 48, 96, 256};
    for (int si = 0; si < 8; si++) {
        uint32_t nv = (uint32_t)(mbs[si] * 1048576 / 16);
        double gb[3], us4 = 0;
        const uint32_t ks[3] = {1, 4, 16};
        for (int ki = 0; ki < 3; ki++) {
            uint32_t wgs = (nv + 256 * ks[ki] - 1) / (256 * ks[ki]);
            double t[41];
            for (int r = 0; r < 41; r++) {
                uint32_t base = (uint32_t)(((uint64_t)r * nv * 3) % (total_v4 - nv));
                begin(); vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, 0);
                disp(base, nv, wgs); vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, 1);
                t[r] = end_us();
            }
            double m = median(t, 41); gb[ki] = nv * 16.0 / (m * 1e3); if (ks[ki] == 4) us4 = m;
        }
        printf("%8.1f %10.1f %10.1f %10.1f %10.1f\n", mbs[si], gb[0], gb[1], gb[2], us4);
    }

    printf("\nchain: 48 dispatches x 12 MB, barrier between each (decode shape), median of 21\n");
    const uint32_t ks[3] = {1, 4, 16};
    for (int ki = 0; ki < 3; ki++) for (int bar = 0; bar < 2; bar++) {
        uint32_t nv = 12u * 1048576 / 16, wgs = (nv + 256 * ks[ki] - 1) / (256 * ks[ki]);
        double t[21];
        for (int r = 0; r < 21; r++) {
            begin(); vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, 0);
            for (int d = 0; d < 48; d++) { if (bar && d) barrier(); disp((uint32_t)(((uint64_t)(r * 48 + d) * nv) % (total_v4 - nv)), nv, wgs); }
            vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, 1); t[r] = end_us();
        }
        double m = median(t, 21);
        printf("  k=%-2u %-10s %8.1f us total  %6.1f us/disp  %6.1f GB/s\n", ks[ki], bar ? "barriers" : "no-barrier", m, m / 48, 48.0 * nv * 16 / (m * 1e3));
    }
    return 0;
}
