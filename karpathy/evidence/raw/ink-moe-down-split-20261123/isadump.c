// Compile one compute .spv on RADV so RADV_DEBUG=shaders dumps only its ISA.
// usage: isadump file.spv nbindings push_bytes [wave32|wave64]
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CK(x) do { VkResult r_ = (x); if (r_) { fprintf(stderr, #x " = %d\n", r_); return 1; } } while (0)

int main(int argc, char **argv) {
    if (argc < 4) return 2;
    FILE *f = fopen(argv[1], "rb"); if (!f) return 3;
    fseek(f, 0, SEEK_END); long n = ftell(f); rewind(f);
    uint32_t *code = malloc(n); fread(code, 1, n, f); fclose(f);
    uint32_t nb = atoi(argv[2]), push = atoi(argv[3]);
    uint32_t wave = argc > 4 ? (strcmp(argv[4], "wave32") ? 64 : 32) : 0;

    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_3};
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app};
    VkInstance inst; CK(vkCreateInstance(&ici, 0, &inst));
    uint32_t pc = 1; VkPhysicalDevice pd; vkEnumeratePhysicalDevices(inst, &pc, &pd);
    float prio = 1;
    VkDeviceQueueCreateInfo q = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueCount = 1, .pQueuePriorities = &prio};
    VkPhysicalDeviceVulkan13Features f13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, .subgroupSizeControl = VK_TRUE, .computeFullSubgroups = VK_TRUE};
    VkPhysicalDeviceVulkan12Features f12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, .pNext = &f13,
        .storageBuffer8BitAccess = 1, .shaderInt8 = 1, .shaderFloat16 = 1};
    VkPhysicalDeviceVulkan11Features f11 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, .pNext = &f12, .storageBuffer16BitAccess = 1};
    VkPhysicalDeviceFeatures2 f2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &f11, .features.shaderInt16 = 1};
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &f2, .queueCreateInfoCount = 1, .pQueueCreateInfos = &q};
    VkDevice dev; CK(vkCreateDevice(pd, &dci, 0, &dev));

    VkDescriptorSetLayoutBinding b[16];
    for (uint32_t i = 0; i < nb; i++)
        b[i] = (VkDescriptorSetLayoutBinding){i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT};
    VkDescriptorSetLayoutCreateInfo dl = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = nb, .pBindings = b};
    VkDescriptorSetLayout dsl; CK(vkCreateDescriptorSetLayout(dev, &dl, 0, &dsl));
    VkPushConstantRange pr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, push};
    VkPipelineLayoutCreateInfo pl = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1, .pSetLayouts = &dsl,
        .pushConstantRangeCount = push ? 1 : 0, .pPushConstantRanges = &pr};
    VkPipelineLayout layout; CK(vkCreatePipelineLayout(dev, &pl, 0, &layout));
    VkShaderModuleCreateInfo sm = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = n, .pCode = code};
    VkShaderModule mod; CK(vkCreateShaderModule(dev, &sm, 0, &mod));
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo ws = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO, .requiredSubgroupSize = wave};
    VkComputePipelineCreateInfo cp = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .pNext = wave ? &ws : 0,
                  .flags = wave ? VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT : 0,
                  .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = mod, .pName = "main"},
        .layout = layout};
    VkPipeline p; CK(vkCreateComputePipelines(dev, 0, 1, &cp, 0, &p));
    fprintf(stderr, "isadump: ok\n");
    return 0;
}
