/* Router-shaped chain: 256-WG producer -> 1-WG consumer, split (two
 * dispatches + two dst-only barriers per link) vs ticket (the producer's last
 * workgroup does the consumer work: one dispatch + one barrier per link). */
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { fprintf(stderr, "%s -> %d\n", #x, r_); exit(1); } } while (0)
enum { N = 2048, W = 256, K = 200, REPS = 15, NV = 2 };
static VkDevice dev;
static int cmp(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }

typedef struct { VkDescriptorSetLayout dsl; VkPipelineLayout pl; VkPipeline pipe; } Pipe;
static Pipe mk(const char *path, int nb, int push) {
    FILE *f = fopen(path, "rb"); if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END); size_t sn = ftell(f); fseek(f, 0, SEEK_SET);
    uint32_t *spv = malloc(sn); if (fread(spv, 1, sn, f) != sn) exit(1); fclose(f);
    VkShaderModule sm; CK(vkCreateShaderModule(dev, &(VkShaderModuleCreateInfo){ .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = sn, .pCode = spv }, NULL, &sm));
    VkDescriptorSetLayoutBinding bd[3];
    for (int i = 0; i < nb; i++) bd[i] = (VkDescriptorSetLayoutBinding){ i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL };
    Pipe p;
    CK(vkCreateDescriptorSetLayout(dev, &(VkDescriptorSetLayoutCreateInfo){ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = nb, .pBindings = bd }, NULL, &p.dsl));
    CK(vkCreatePipelineLayout(dev, &(VkPipelineLayoutCreateInfo){ .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1, .pSetLayouts = &p.dsl,
        .pushConstantRangeCount = 1, .pPushConstantRanges = &(VkPushConstantRange){ VK_SHADER_STAGE_COMPUTE_BIT, 0, push } }, NULL, &p.pl));
    CK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &(VkComputePipelineCreateInfo){ .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = sm, .pName = "main" }, .layout = p.pl }, NULL, &p.pipe));
    return p;
}

int main(void) {
    VkInstance inst; CK(vkCreateInstance(&(VkInstanceCreateInfo){ .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &(VkApplicationInfo){ .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_1 } }, NULL, &inst));
    uint32_t n = 1; VkPhysicalDevice pd; CK(vkEnumeratePhysicalDevices(inst, &n, &pd));
    VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(pd, &props);
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    float prio = 1.0f;
    CK(vkCreateDevice(pd, &(VkDeviceCreateInfo){ .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &(VkDeviceQueueCreateInfo){ .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = 0, .queueCount = 1, .pQueuePriorities = &prio } }, NULL, &dev));
    VkQueue q; vkGetDeviceQueue(dev, 0, 0, &q);
    Pipe pr = mk("rprod.spv", 3, 8), pc = mk("rcons.spv", 2, 4);
    VkDescriptorPool dp; CK(vkCreateDescriptorPool(dev, &(VkDescriptorPoolCreateInfo){ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 4, .poolSizeCount = 1,
        .pPoolSizes = &(VkDescriptorPoolSize){ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8 } }, NULL, &dp));
    VkCommandPool cp; CK(vkCreateCommandPool(dev, &(VkCommandPoolCreateInfo){ .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT }, NULL, &cp));
    VkCommandBuffer cb; CK(vkAllocateCommandBuffers(dev, &(VkCommandBufferAllocateInfo){ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = cp, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1 }, &cb));
    VkQueryPool qp; CK(vkCreateQueryPool(dev, &(VkQueryPoolCreateInfo){ .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = 2 }, NULL, &qp));
    uint32_t t = UINT32_MAX;
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) if (mp.memoryTypes[i].propertyFlags == 0x07) { t = i; break; }
    if (t == UINT32_MAX) return 1;
    VkBuffer buf[3]; VkDeviceMemory mem[3]; uint32_t *map[3]; const VkDeviceSize sz[3] = { N * 4, W * 4, 4 };
    for (int b = 0; b < 3; b++) {
        CK(vkCreateBuffer(dev, &(VkBufferCreateInfo){ .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = sz[b], .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT }, NULL, &buf[b]));
        VkMemoryRequirements req; vkGetBufferMemoryRequirements(dev, buf[b], &req);
        CK(vkAllocateMemory(dev, &(VkMemoryAllocateInfo){ .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = req.size, .memoryTypeIndex = t }, NULL, &mem[b]));
        CK(vkBindBufferMemory(dev, buf[b], mem[b], 0));
        CK(vkMapMemory(dev, mem[b], 0, sz[b], 0, (void **)&map[b]));
    }
    VkDescriptorSet ds[2]; VkDescriptorSetLayout ls[2] = { pr.dsl, pc.dsl };
    CK(vkAllocateDescriptorSets(dev, &(VkDescriptorSetAllocateInfo){ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = dp, .descriptorSetCount = 2, .pSetLayouts = ls }, ds));
    for (int s = 0; s < 2; s++) for (int b = 0; b < (s ? 2 : 3); b++) {
        VkDescriptorBufferInfo bi = { buf[b], 0, VK_WHOLE_SIZE };
        vkUpdateDescriptorSets(dev, 1, &(VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds[s], .dstBinding = b,
            .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bi }, 0, NULL);
    }
    VkMemoryBarrier mb = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = 0, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT };
    const char *vname[NV] = { "split 2 disp", "ticket 1 disp" };
    double ms[NV][REPS]; int fails[NV] = { 0 };
    for (int r = 0; r < REPS; r++) for (int vv = 0; vv < NV; vv++) {
        int v = (vv + r) % NV;
        memset(map[0], 0, N * 4); memset(map[1], 0, W * 4); map[2][0] = 0;
        CK(vkResetCommandBuffer(cb, 0));
        CK(vkBeginCommandBuffer(cb, &(VkCommandBufferBeginInfo){ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO }));
        vkCmdResetQueryPool(cb, qp, 0, 2);
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, 0);
        uint32_t ppush[2] = { N, (uint32_t)v }, cpush = N;
        for (int k = 0; k < K; k++) {
            if (k > 0) vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pr.pipe);
            vkCmdPushConstants(cb, pr.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, ppush);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pr.pl, 0, 1, &ds[0], 0, NULL);
            vkCmdDispatch(cb, W, 1, 1);
            if (v == 0) {
                vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
                vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pc.pipe);
                vkCmdPushConstants(cb, pc.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &cpush);
                vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pc.pl, 0, 1, &ds[1], 0, NULL);
                vkCmdDispatch(cb, 1, 1, 1);
            }
        }
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, 1);
        VkMemoryBarrier hb = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_HOST_READ_BIT };
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &hb, 0, NULL, 0, NULL);
        CK(vkEndCommandBuffer(cb));
        CK(vkQueueSubmit(q, 1, &(VkSubmitInfo){ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb }, VK_NULL_HANDLE));
        CK(vkQueueWaitIdle(q));
        uint64_t ts[2];
        CK(vkGetQueryPoolResults(dev, qp, 0, 2, sizeof ts, ts, 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
        ms[v][r] = (double)(ts[1] - ts[0]) * props.limits.timestampPeriod * 1e-6;
        for (int i = 0; i < N; i++) if (map[0][i] != K) { fails[v]++; break; }
    }
    printf("router-shaped chain: %d links, producer %d WGs -> consumer 1 WG, %d reps\n", K, W, REPS);
    for (int v = 0; v < NV; v++) {
        qsort(ms[v], REPS, sizeof(double), cmp);
        printf("  %-14s median %7.3f ms  min %7.3f  max %7.3f  = %6.2f us/link  wrong %d/%d\n",
               vname[v], ms[v][REPS / 2], ms[v][0], ms[v][REPS - 1], 1e3 * ms[v][REPS / 2] / K, fails[v], REPS);
    }
    return 0;
}
