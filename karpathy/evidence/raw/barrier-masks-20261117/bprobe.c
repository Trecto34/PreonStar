/* Cost of one hazard barrier on the BC-250 GFX queue, by barrier form.
 * A chain of K dependent decode-shaped dispatches (32 WGs x 64 threads, each
 * reading a spread of the whole 2048-word input), one barrier between links:
 *   A  engine today: COMPUTE|TRANSFER, SHADER_WRITE|TRANSFER_WRITE -> SHADER_READ|WRITE
 *   B  compute-only: COMPUTE,          SHADER_WRITE                -> SHADER_READ|WRITE
 *   C  dst-only:     COMPUTE,          0                           -> SHADER_READ|WRITE
 *   D  no barrier (timing floor only; numerics may be wrong)
 * Every rep checks dst == K, so a form that loses a write shows up as FAIL. */
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { fprintf(stderr, "%s -> %d\n", #x, r_); exit(1); } } while (0)
enum { N = 2048, K = 400, REPS = 15, NV = 4 };

static int cmp(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }

int main(int argc, char **argv) {
    VkInstance inst; CK(vkCreateInstance(&(VkInstanceCreateInfo){ .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &(VkApplicationInfo){ .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_1 } }, NULL, &inst));
    uint32_t n = 1; VkPhysicalDevice pd; CK(vkEnumeratePhysicalDevices(inst, &n, &pd));
    VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(pd, &props);
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    float prio = 1.0f; VkDevice dev;
    CK(vkCreateDevice(pd, &(VkDeviceCreateInfo){ .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &(VkDeviceQueueCreateInfo){ .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = 0, .queueCount = 1, .pQueuePriorities = &prio } }, NULL, &dev));
    VkQueue q; vkGetDeviceQueue(dev, 0, 0, &q);

    FILE *f = fopen(argc > 1 ? argv[1] : "chain.spv", "rb"); if (!f) { perror("spv"); return 1; }
    fseek(f, 0, SEEK_END); size_t sn = ftell(f); fseek(f, 0, SEEK_SET);
    uint32_t *spv = malloc(sn); if (fread(spv, 1, sn, f) != sn) return 1; fclose(f);
    VkShaderModule sm; CK(vkCreateShaderModule(dev, &(VkShaderModuleCreateInfo){ .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = sn, .pCode = spv }, NULL, &sm));
    VkDescriptorSetLayoutBinding bd[2] = { { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
                                           { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL } };
    VkDescriptorSetLayout dsl; CK(vkCreateDescriptorSetLayout(dev, &(VkDescriptorSetLayoutCreateInfo){ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 2, .pBindings = bd }, NULL, &dsl));
    VkPipelineLayout pl; CK(vkCreatePipelineLayout(dev, &(VkPipelineLayoutCreateInfo){ .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1, .pSetLayouts = &dsl,
        .pushConstantRangeCount = 1, .pPushConstantRanges = &(VkPushConstantRange){ VK_SHADER_STAGE_COMPUTE_BIT, 0, 4 } }, NULL, &pl));
    VkPipeline pipe; CK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &(VkComputePipelineCreateInfo){ .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = sm, .pName = "main" }, .layout = pl }, NULL, &pipe));
    VkDescriptorPool dp; CK(vkCreateDescriptorPool(dev, &(VkDescriptorPoolCreateInfo){ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 4, .poolSizeCount = 1,
        .pPoolSizes = &(VkDescriptorPoolSize){ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8 } }, NULL, &dp));
    VkCommandPool cp; CK(vkCreateCommandPool(dev, &(VkCommandPoolCreateInfo){ .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT }, NULL, &cp));
    VkCommandBuffer cb; CK(vkAllocateCommandBuffers(dev, &(VkCommandBufferAllocateInfo){ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = cp, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1 }, &cb));
    VkQueryPool qp; CK(vkCreateQueryPool(dev, &(VkQueryPoolCreateInfo){ .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = 2 }, NULL, &qp));

    /* The engine's activation memory (HOST_CACHED) and its device-local weight heap. */
    const uint32_t want[2] = { 0x0e, 0x07 };
    const char *vname[NV] = { "A engine", "B compute-only", "C dst-only", "D none" };
    for (int wi = 0; wi < 2; wi++) {
        uint32_t t = UINT32_MAX;
        for (uint32_t i = 0; i < mp.memoryTypeCount; i++) if (mp.memoryTypes[i].propertyFlags == want[wi]) { t = i; break; }
        if (t == UINT32_MAX) { printf("no memory type with flags 0x%02x\n", want[wi]); continue; }
        VkBuffer buf[2]; VkDeviceMemory mem[2]; uint32_t *map[2]; VkDescriptorSet ds[2];
        for (int b = 0; b < 2; b++) {
            CK(vkCreateBuffer(dev, &(VkBufferCreateInfo){ .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = N * 4,
                .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT }, NULL, &buf[b]));
            VkMemoryRequirements req; vkGetBufferMemoryRequirements(dev, buf[b], &req);
            CK(vkAllocateMemory(dev, &(VkMemoryAllocateInfo){ .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = req.size, .memoryTypeIndex = t }, NULL, &mem[b]));
            CK(vkBindBufferMemory(dev, buf[b], mem[b], 0));
            CK(vkMapMemory(dev, mem[b], 0, N * 4, 0, (void **)&map[b]));
        }
        for (int s = 0; s < 2; s++) {
            CK(vkAllocateDescriptorSets(dev, &(VkDescriptorSetAllocateInfo){ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = dp, .descriptorSetCount = 1, .pSetLayouts = &dsl }, &ds[s]));
            VkDescriptorBufferInfo bi[2] = { { buf[s], 0, VK_WHOLE_SIZE }, { buf[!s], 0, VK_WHOLE_SIZE } };
            VkWriteDescriptorSet w[2] = {
                { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds[s], .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bi[0] },
                { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds[s], .dstBinding = 1, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bi[1] } };
            vkUpdateDescriptorSets(dev, 2, w, 0, NULL);
        }
        double ms[NV][REPS]; int fails[NV] = { 0 };
        for (int r = 0; r < REPS; r++) {
            for (int vv = 0; vv < NV; vv++) {
                int v = (vv + r) % NV;   /* rotate the order every rep */
                memset(map[0], 0, N * 4); memset(map[1], 0, N * 4);
                CK(vkResetCommandBuffer(cb, 0));
                CK(vkBeginCommandBuffer(cb, &(VkCommandBufferBeginInfo){ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO }));
                vkCmdResetQueryPool(cb, qp, 0, 2);
                vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
                uint32_t nn = N; vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &nn);
                vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, 0);
                for (int k = 0; k < K; k++) {
                    if (k > 0 && v != 3) {
                        VkMemoryBarrier mb = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                            .srcAccessMask = v == 0 ? VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT
                                           : v == 1 ? VK_ACCESS_SHADER_WRITE_BIT : 0,
                            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT };
                        vkCmdPipelineBarrier(cb, v == 0 ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT
                                                        : VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
                    }
                    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds[k & 1], 0, NULL);
                    vkCmdDispatch(cb, N / 64, 1, 1);
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
                const uint32_t *out = map[K & 1];   /* link k writes buf[!(k&1)]; last link k=K-1 */
                for (int i = 0; i < N; i++) if (out[i] != K) { fails[v]++; break; }
            }
        }
        printf("memory type %u flags 0x%02x, %d links x %d WGs, %d reps\n", t, want[wi], K, N / 64, REPS);
        double med[NV];
        for (int v = 0; v < NV; v++) { qsort(ms[v], REPS, sizeof(double), cmp); med[v] = ms[v][REPS / 2]; }
        for (int v = 0; v < NV; v++)
            printf("  %-15s median %7.3f ms  min %7.3f  max %7.3f  = %6.2f us/link  barrier %6.2f us  wrong %d/%d\n",
                   vname[v], med[v], ms[v][0], ms[v][REPS - 1], 1e3 * med[v] / K,
                   1e3 * (med[v] - med[3]) / (K - 1), fails[v], REPS);
        for (int b = 0; b < 2; b++) { vkUnmapMemory(dev, mem[b]); vkDestroyBuffer(dev, buf[b], NULL); vkFreeMemory(dev, mem[b], NULL); }
        CK(vkResetDescriptorPool(dev, dp, 0));
    }
    return 0;
}
