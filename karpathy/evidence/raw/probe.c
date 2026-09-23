/* GPU streaming-read ceiling on the BC-250: one 1 GiB buffer per memory type,
 * uvec4 grid-stride reads, timestamp-query timing, best of N reps. */
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { fprintf(stderr, "%s -> %d\n", #x, r_); exit(1); } } while (0)

static VkDevice dev;
static VkPhysicalDevice pd;
static VkPhysicalDeviceMemoryProperties mp;

static unsigned char *slurp(const char *p, size_t *n) {
    FILE *f = fopen(p, "rb"); if (!f) { perror(p); exit(1); }
    fseek(f, 0, SEEK_END); *n = ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char *b = malloc(*n); fread(b, 1, *n, f); fclose(f); return b;
}

int main(int argc, char **argv) {
    const VkDeviceSize bytes = (VkDeviceSize)1 << 30;
    const int reps = 10;
    VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &(VkApplicationInfo){ .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_1 } };
    VkInstance inst; CK(vkCreateInstance(&ici, NULL, &inst));
    uint32_t n = 1; CK(vkEnumeratePhysicalDevices(inst, &n, &pd));
    VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(pd, &props);
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    float ts_period = props.limits.timestampPeriod;
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = 0, .queueCount = 1, .pQueuePriorities = &prio };
    VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci };
    CK(vkCreateDevice(pd, &dci, NULL, &dev));
    VkQueue q; vkGetDeviceQueue(dev, 0, 0, &q);

    size_t spv_n; unsigned char *spv = slurp(argc > 1 ? argv[1] : "stream.spv", &spv_n);
    VkShaderModule sm; CK(vkCreateShaderModule(dev, &(VkShaderModuleCreateInfo){ .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = spv_n, .pCode = (uint32_t *)spv }, NULL, &sm));
    VkDescriptorSetLayoutBinding b[2] = {
        { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
        { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL } };
    VkDescriptorSetLayout dsl; CK(vkCreateDescriptorSetLayout(dev, &(VkDescriptorSetLayoutCreateInfo){ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 2, .pBindings = b }, NULL, &dsl));
    VkPushConstantRange pcr = { VK_SHADER_STAGE_COMPUTE_BIT, 0, 8 };
    VkPipelineLayout pl; CK(vkCreatePipelineLayout(dev, &(VkPipelineLayoutCreateInfo){ .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1, .pSetLayouts = &dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr }, NULL, &pl));
    VkPipeline pipe; CK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &(VkComputePipelineCreateInfo){ .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = sm, .pName = "main" }, .layout = pl }, NULL, &pipe));
    VkDescriptorPool dp; CK(vkCreateDescriptorPool(dev, &(VkDescriptorPoolCreateInfo){ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 1, .poolSizeCount = 1,
        .pPoolSizes = &(VkDescriptorPoolSize){ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 } }, NULL, &dp));
    VkCommandPool cp; CK(vkCreateCommandPool(dev, &(VkCommandPoolCreateInfo){ .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = 0 }, NULL, &cp));
    VkQueryPool qp; CK(vkCreateQueryPool(dev, &(VkQueryPoolCreateInfo){ .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = 2 }, NULL, &qp));

    /* output buffer: any type */
    VkBuffer ob; CK(vkCreateBuffer(dev, &(VkBufferCreateInfo){ .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = 256, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT }, NULL, &ob));
    VkMemoryRequirements oreq; vkGetBufferMemoryRequirements(dev, ob, &oreq);
    VkDeviceMemory om; CK(vkAllocateMemory(dev, &(VkMemoryAllocateInfo){ .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = oreq.size, .memoryTypeIndex = __builtin_ctz(oreq.memoryTypeBits) }, NULL, &om));
    CK(vkBindBufferMemory(dev, ob, om, 0));

    int types[] = { 0, 3, 5 };
    uint32_t groups_list[] = { 320, 640, 1280, 2560, 5120 };
    for (unsigned ti = 0; ti < sizeof types / sizeof *types; ti++) {
        int t = types[ti];
        VkBuffer buf; CK(vkCreateBuffer(dev, &(VkBufferCreateInfo){ .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = bytes, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT }, NULL, &buf));
        VkMemoryRequirements req; vkGetBufferMemoryRequirements(dev, buf, &req);
        if (!(req.memoryTypeBits & (1u << t))) { printf("type %d unsupported\n", t); continue; }
        VkDeviceMemory mem;
        if (vkAllocateMemory(dev, &(VkMemoryAllocateInfo){ .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = req.size, .memoryTypeIndex = t }, NULL, &mem) != VK_SUCCESS) { printf("type %d alloc failed\n", t); continue; }
        CK(vkBindBufferMemory(dev, buf, mem, 0));
        VkDescriptorSet ds; CK(vkAllocateDescriptorSets(dev, &(VkDescriptorSetAllocateInfo){ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = dp, .descriptorSetCount = 1, .pSetLayouts = &dsl }, &ds));
        VkDescriptorBufferInfo bi[2] = { { buf, 0, VK_WHOLE_SIZE }, { ob, 0, VK_WHOLE_SIZE } };
        VkWriteDescriptorSet w[2] = {
            { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bi[0] },
            { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = 1, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bi[1] } };
        vkUpdateDescriptorSets(dev, 2, w, 0, NULL);
        VkCommandBuffer cb; CK(vkAllocateCommandBuffers(dev, &(VkCommandBufferAllocateInfo){ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = cp, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1 }, &cb));
        /* fill once so pages are resident */
        CK(vkBeginCommandBuffer(cb, &(VkCommandBufferBeginInfo){ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO }));
        vkCmdFillBuffer(cb, buf, 0, VK_WHOLE_SIZE, 0x01020304u);
        CK(vkEndCommandBuffer(cb));
        CK(vkQueueSubmit(q, 1, &(VkSubmitInfo){ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb }, VK_NULL_HANDLE));
        CK(vkQueueWaitIdle(q));
        for (unsigned gi = 0; gi < sizeof groups_list / sizeof *groups_list; gi++) {
            double best = 1e30;
            for (int r = 0; r < reps; r++) {
                CK(vkResetCommandBuffer(cb, 0));
                CK(vkBeginCommandBuffer(cb, &(VkCommandBufferBeginInfo){ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO }));
                vkCmdResetQueryPool(cb, qp, 0, 2);
                vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
                vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
                uint32_t push[2] = { (uint32_t)(bytes / 16), 1 };
                vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, push);
                vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, 0);
                vkCmdDispatch(cb, groups_list[gi], 1, 1);
                vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, 1);
                CK(vkEndCommandBuffer(cb));
                CK(vkQueueSubmit(q, 1, &(VkSubmitInfo){ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb }, VK_NULL_HANDLE));
                CK(vkQueueWaitIdle(q));
                uint64_t ts[2];
                CK(vkGetQueryPoolResults(dev, qp, 0, 2, sizeof ts, ts, 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
                double ms = (double)(ts[1] - ts[0]) * ts_period * 1e-6;
                if (ms < best) best = ms;
            }
            printf("type %d flags 0x%02x groups %5u: best %.3f ms  %.1f GB/s\n", t, mp.memoryTypes[t].propertyFlags,
                   groups_list[gi], best, (double)bytes / (best * 1e-3) / 1e9);
        }
        vkFreeCommandBuffers(dev, cp, 1, &cb);
        CK(vkResetDescriptorPool(dev, dp, 0));
        vkDestroyBuffer(dev, buf, NULL);
        vkFreeMemory(dev, mem, NULL);
    }
    return 0;
}
