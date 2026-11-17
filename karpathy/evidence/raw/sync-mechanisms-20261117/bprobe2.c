/* Per-link cost of three ways to chain K dependent decode-shaped steps
 * (32 WGs x 64 threads, each reading a spread of a 2048-word vector):
 *   C  one dispatch per link + the engine's dst-only barrier (main today)
 *   F  one dispatch per link, NO barrier; workgroups spin on a per-link
 *      counter the previous link's workgroups bump (counter-chaining, lever 2)
 *   G  ONE dispatch for all links, software grid barrier between links
 *      (persistent kernel, lever 3)
 *   D  one dispatch per link, no sync at all (floor; numerics wrong)
 * Every rep checks the result == K. */
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { fprintf(stderr, "%s -> %d\n", #x, r_); exit(1); } } while (0)
enum { N = 2048, G = N / 64, K = 400, REPS = 15, NV = 4 };
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
    Pipe pc = mk("chain.spv", 2, 4), pf = mk("chain_flag.spv", 3, 12), pp = mk("chain_persist.spv", 3, 12);
    VkDescriptorPool dp; CK(vkCreateDescriptorPool(dev, &(VkDescriptorPoolCreateInfo){ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 8, .poolSizeCount = 1,
        .pPoolSizes = &(VkDescriptorPoolSize){ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 24 } }, NULL, &dp));
    VkCommandPool cp; CK(vkCreateCommandPool(dev, &(VkCommandPoolCreateInfo){ .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT }, NULL, &cp));
    VkCommandBuffer cb; CK(vkAllocateCommandBuffers(dev, &(VkCommandBufferAllocateInfo){ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = cp, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1 }, &cb));
    VkQueryPool qp; CK(vkCreateQueryPool(dev, &(VkQueryPoolCreateInfo){ .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = 2 }, NULL, &qp));

    uint32_t t = UINT32_MAX;   /* device-local, host-visible: the engine's weight heap type */
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) if (mp.memoryTypes[i].propertyFlags == 0x07) { t = i; break; }
    if (t == UINT32_MAX) { printf("no 0x07 memory type\n"); return 1; }
    VkBuffer buf[3]; VkDeviceMemory mem[3]; uint32_t *map[3]; const VkDeviceSize sz[3] = { N * 4, N * 4, K * 4 };
    for (int b = 0; b < 3; b++) {
        CK(vkCreateBuffer(dev, &(VkBufferCreateInfo){ .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = sz[b], .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT }, NULL, &buf[b]));
        VkMemoryRequirements req; vkGetBufferMemoryRequirements(dev, buf[b], &req);
        CK(vkAllocateMemory(dev, &(VkMemoryAllocateInfo){ .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = req.size, .memoryTypeIndex = t }, NULL, &mem[b]));
        CK(vkBindBufferMemory(dev, buf[b], mem[b], 0));
        CK(vkMapMemory(dev, mem[b], 0, sz[b], 0, (void **)&map[b]));
    }
    /* sets: chain 0->1, chain 1->0, flag 0->1, flag 1->0, persist */
    VkDescriptorSet ds[5]; VkDescriptorSetLayout ls[5] = { pc.dsl, pc.dsl, pf.dsl, pf.dsl, pp.dsl };
    CK(vkAllocateDescriptorSets(dev, &(VkDescriptorSetAllocateInfo){ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = dp, .descriptorSetCount = 5, .pSetLayouts = ls }, ds));
    const int src[5] = { 0, 1, 0, 1, 0 }, nb[5] = { 2, 2, 3, 3, 3 };
    for (int s = 0; s < 5; s++) {
        VkDescriptorBufferInfo bi[3] = { { buf[src[s]], 0, VK_WHOLE_SIZE }, { buf[!src[s]], 0, VK_WHOLE_SIZE }, { buf[2], 0, VK_WHOLE_SIZE } };
        for (int b = 0; b < nb[s]; b++)
            vkUpdateDescriptorSets(dev, 1, &(VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds[s], .dstBinding = b,
                .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bi[b] }, 0, NULL);
    }
    const char *vname[NV] = { "C dst-only bar", "F counter-chain", "G persistent", "D none" };
    double ms[NV][REPS]; int fails[NV] = { 0 };
    for (int r = 0; r < REPS; r++) for (int vv = 0; vv < NV; vv++) {
        int v = (vv + r) % NV;
        memset(map[0], 0, N * 4); memset(map[1], 0, N * 4); memset(map[2], 0, K * 4);
        CK(vkResetCommandBuffer(cb, 0));
        CK(vkBeginCommandBuffer(cb, &(VkCommandBufferBeginInfo){ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO }));
        vkCmdResetQueryPool(cb, qp, 0, 2);
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, 0);
        if (v == 2) {
            uint32_t push[3] = { N, K, G };
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pp.pipe);
            vkCmdPushConstants(cb, pp.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 12, push);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pp.pl, 0, 1, &ds[4], 0, NULL);
            vkCmdDispatch(cb, G, 1, 1);
        } else {
            Pipe *p = v == 1 ? &pf : &pc;
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p->pipe);
            for (int k = 0; k < K; k++) {
                if (k > 0 && v == 0) {
                    VkMemoryBarrier mb = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = 0,
                                           .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT };
                    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
                }
                uint32_t push[3] = { N, (uint32_t)k, G };
                vkCmdPushConstants(cb, p->pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, v == 1 ? 12 : 4, push);
                vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p->pl, 0, 1, &ds[(v == 1 ? 2 : 0) + (k & 1)], 0, NULL);
                vkCmdDispatch(cb, G, 1, 1);
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
        for (int i = 0; i < N; i++) if (map[0][i] != K) { fails[v]++; break; }   /* K even: the last link writes buf 0 */
    }
    printf("memory type %u flags 0x07, %d links x %d WGs, %d reps\n", t, K, G, REPS);
    for (int v = 0; v < NV; v++) {
        qsort(ms[v], REPS, sizeof(double), cmp);
        printf("  %-16s median %7.3f ms  min %7.3f  max %7.3f  = %6.2f us/link  wrong %d/%d\n",
               vname[v], ms[v][REPS / 2], ms[v][0], ms[v][REPS - 1], 1e3 * ms[v][REPS / 2] / K, fails[v], REPS);
    }
    return 0;
}
