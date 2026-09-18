/* Emit ACO ISA for a compute .spv via VK_KHR_pipeline_executable_properties.
 * Separate short-lived process: if RADV faults on the IR query, only this dies.
 * Descriptor layout is taken from the SPIR-V itself so it fits any shader.
 *   cc -O2 -o shader_isa shader_isa.c -lvulkan
 *   ./shader_isa <shader.spv> [push_constant_bytes]
 */
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(c) do { VkResult r_=(c); if(r_!=VK_SUCCESS){fprintf(stderr,"%s -> %d\n",#c,r_);exit(1);} } while(0)

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc < 2) { fprintf(stderr, "usage: %s shader.spv [pc_bytes]\n", argv[0]); return 2; }
    uint32_t pc_bytes = argc > 2 ? (uint32_t)atoi(argv[2]) : 32;

    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 1; }
    fseek(f, 0, SEEK_END); long len = ftell(f); rewind(f);
    uint32_t *spv = malloc((size_t)len);
    if (fread(spv, 1, (size_t)len, f) != (size_t)len) return 1;
    fclose(f);

    /* Count descriptor bindings by scanning OpDecorate ... Binding. */
    uint32_t nbind = 0;
    for (long i = 5; i * 4 < len; ) {
        uint32_t w = spv[i], op = w & 0xffff, wc = w >> 16;
        if (!wc) break;
        if (op == 71 /* OpDecorate */ && wc >= 4 && spv[i + 2] == 33 /* Binding */) {
            uint32_t b = spv[i + 3];
            if (b + 1 > nbind) nbind = b + 1;
        }
        i += wc;
    }
    if (!nbind) nbind = 4;
    fprintf(stderr, "[dbg] %u descriptor bindings, %u push bytes\n", nbind, pc_bytes);

    VkApplicationInfo app = {.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName="shader-isa", .apiVersion=VK_API_VERSION_1_2};
    VkInstanceCreateInfo ici = {.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo=&app};
    VkInstance inst; CHECK(vkCreateInstance(&ici, NULL, &inst));
    uint32_t n=0; CHECK(vkEnumeratePhysicalDevices(inst,&n,NULL));
    VkPhysicalDevice *devs = calloc(n, sizeof(*devs));
    CHECK(vkEnumeratePhysicalDevices(inst,&n,devs));
    VkPhysicalDevice phys = devs[0]; free(devs);

    uint32_t qn=0; vkGetPhysicalDeviceQueueFamilyProperties(phys,&qn,NULL);
    VkQueueFamilyProperties *qp=calloc(qn,sizeof(*qp));
    vkGetPhysicalDeviceQueueFamilyProperties(phys,&qn,qp);
    uint32_t fam=0; for(uint32_t i=0;i<qn;i++) if(qp[i].queueFlags&VK_QUEUE_COMPUTE_BIT){fam=i;break;}
    free(qp);

    float prio=1.0f;
    VkDeviceQueueCreateInfo qci={.sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex=fam,.queueCount=1,.pQueuePriorities=&prio};
    VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR pepf={
        .sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR,
        .pipelineExecutableInfo=VK_TRUE};
    const char *ext[] = { "VK_KHR_pipeline_executable_properties" };
    VkDeviceCreateInfo dci={.sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext=&pepf,.queueCreateInfoCount=1,.pQueueCreateInfos=&qci,
        .enabledExtensionCount=1,.ppEnabledExtensionNames=ext};
    VkDevice dev; CHECK(vkCreateDevice(phys,&dci,NULL,&dev));

    VkShaderModuleCreateInfo smci={.sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize=(size_t)len,.pCode=spv};
    VkShaderModule sm; CHECK(vkCreateShaderModule(dev,&smci,NULL,&sm));
    free(spv);

    VkDescriptorSetLayoutBinding *binds = calloc(nbind, sizeof(*binds));
    for (uint32_t i=0;i<nbind;i++) binds[i]=(VkDescriptorSetLayoutBinding){.binding=i,
        .descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.descriptorCount=1,
        .stageFlags=VK_SHADER_STAGE_COMPUTE_BIT};
    VkDescriptorSetLayoutCreateInfo slci={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount=nbind,.pBindings=binds};
    VkDescriptorSetLayout sl; CHECK(vkCreateDescriptorSetLayout(dev,&slci,NULL,&sl));
    VkPushConstantRange pcr={.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,.offset=0,.size=pc_bytes};
    VkPipelineLayoutCreateInfo plci={.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount=1,.pSetLayouts=&sl,.pushConstantRangeCount=1,.pPushConstantRanges=&pcr};
    VkPipelineLayout pl; CHECK(vkCreatePipelineLayout(dev,&plci,NULL,&pl));

    VkPipelineShaderStageCreateInfo stage={.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage=VK_SHADER_STAGE_COMPUTE_BIT,.module=sm,.pName="main"};
    VkComputePipelineCreateInfo pci={.sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage=stage,.layout=pl,
        .flags=VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR |
               VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR};
    VkPipeline pipe; CHECK(vkCreateComputePipelines(dev,VK_NULL_HANDLE,1,&pci,NULL,&pipe));
    fprintf(stderr, "[dbg] pipeline built with CAPTURE_INTERNAL_REPRESENTATIONS\n");

    PFN_vkGetPipelineExecutablePropertiesKHR getProps =
        (PFN_vkGetPipelineExecutablePropertiesKHR)vkGetDeviceProcAddr(dev,"vkGetPipelineExecutablePropertiesKHR");
    PFN_vkGetPipelineExecutableInternalRepresentationsKHR getIR =
        (PFN_vkGetPipelineExecutableInternalRepresentationsKHR)vkGetDeviceProcAddr(dev,"vkGetPipelineExecutableInternalRepresentationsKHR");
    if(!getProps||!getIR){fprintf(stderr,"entry points missing\n");return 4;}

    VkPipelineInfoKHR pi={.sType=VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR,.pipeline=pipe};
    uint32_t pn=0; CHECK(getProps(dev,&pi,&pn,NULL));
    VkPipelineExecutablePropertiesKHR *props=calloc(pn,sizeof(*props));
    for(uint32_t i=0;i<pn;i++) props[i].sType=VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR;
    CHECK(getProps(dev,&pi,&pn,props));

    for(uint32_t i=0;i<pn;i++){
        VkPipelineExecutableInfoKHR ei={.sType=VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR,
            .pipeline=pipe,.executableIndex=i};
        uint32_t rn=0;
        VkResult r = getIR(dev,&ei,&rn,NULL);
        fprintf(stderr,"[dbg] exec %u '%s': IR count query -> %d, n=%u\n",i,props[i].name,r,rn);
        if(r!=VK_SUCCESS || !rn) continue;
        VkPipelineExecutableInternalRepresentationKHR *irs=calloc(rn,sizeof(*irs));
        for(uint32_t j=0;j<rn;j++) irs[j].sType=VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INTERNAL_REPRESENTATION_KHR;
        /* First pass fills names + dataSize; second pass fills pData. */
        r = getIR(dev,&ei,&rn,irs);
        fprintf(stderr,"[dbg] sizes query -> %d\n",r);
        for(uint32_t j=0;j<rn;j++){
            fprintf(stderr,"[dbg]   ir %u '%s' text=%d size=%zu\n",
                    j,irs[j].name,irs[j].isText,irs[j].dataSize);
            if(!irs[j].dataSize) continue;
            irs[j].pData = malloc(irs[j].dataSize);
        }
        r = getIR(dev,&ei,&rn,irs);
        fprintf(stderr,"[dbg] data query -> %d\n",r);
        for(uint32_t j=0;j<rn;j++){
            if(!irs[j].pData) continue;
            printf("\n===== executable %u '%s' / IR '%s' (%s) =====\n%s\n",
                   i,props[i].name,irs[j].name,irs[j].description,
                   irs[j].isText ? (char*)irs[j].pData : "(binary)");
        }
    }
    return 0;
}
