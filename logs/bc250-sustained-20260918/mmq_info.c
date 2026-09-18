/* Bounded, unprivileged shader introspection: create the production shader as
 * a Vulkan compute pipeline and ask RADV for its executable properties and
 * internal representations through VK_KHR_pipeline_executable_properties.
 * No RADV_DEBUG / ACO_DEBUG, so it cannot wedge the driver.  Build:
 *   cc -O2 -o mmq_info mmq_info.c -lvulkan
 * Run:
 *   ./mmq_info vulkan/dense_extra_mmq_q2_0.spv
 */
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(c) do { VkResult r_=(c); if(r_!=VK_SUCCESS){fprintf(stderr,"%s -> %d\n",#c,r_);exit(1);} } while(0)

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc != 2) { fprintf(stderr, "usage: %s shader.spv\n", argv[0]); return 2; }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 1; }
    fseek(f, 0, SEEK_END); long len = ftell(f); rewind(f);
    uint32_t *spv = malloc((size_t)len);
    if (fread(spv, 1, (size_t)len, f) != (size_t)len) return 1;
    fclose(f);

    VkApplicationInfo app = {.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName="mmq-info", .apiVersion=VK_API_VERSION_1_2};
    VkInstanceCreateInfo ici = {.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo=&app};
    VkInstance inst; CHECK(vkCreateInstance(&ici, NULL, &inst));

    uint32_t n=0; CHECK(vkEnumeratePhysicalDevices(inst,&n,NULL));
    VkPhysicalDevice *devs = calloc(n, sizeof(*devs));
    CHECK(vkEnumeratePhysicalDevices(inst,&n,devs));
    VkPhysicalDevice phys = devs[0]; free(devs);

    uint32_t ec=0; CHECK(vkEnumerateDeviceExtensionProperties(phys,NULL,&ec,NULL));
    VkExtensionProperties *eps = calloc(ec, sizeof(*eps));
    CHECK(vkEnumerateDeviceExtensionProperties(phys,NULL,&ec,eps));
    int has_pep=0; for (uint32_t i=0;i<ec;i++) if(!strcmp(eps[i].extensionName,"VK_KHR_pipeline_executable_properties")) has_pep=1;
    free(eps);
    printf("VK_KHR_pipeline_executable_properties: %s\n", has_pep?"yes":"NO");
    if (!has_pep) return 3;

    uint32_t qn=0; vkGetPhysicalDeviceQueueFamilyProperties(phys,&qn,NULL);
    VkQueueFamilyProperties *qp=calloc(qn,sizeof(*qp));
    vkGetPhysicalDeviceQueueFamilyProperties(phys,&qn,qp);
    uint32_t fam=UINT32_MAX; for(uint32_t i=0;i<qn;i++) if(qp[i].queueFlags&VK_QUEUE_COMPUTE_BIT){fam=i;break;}
    free(qp);

    float prio=1.0f;
    VkDeviceQueueCreateInfo qci={.sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex=fam,.queueCount=1,.pQueuePriorities=&prio};
    VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR pepf={
        .sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR,
        .pipelineExecutableInfo=VK_TRUE};
    const char *enenames[] = { "VK_KHR_pipeline_executable_properties" };
    VkDeviceCreateInfo dci={.sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext=&pepf,.queueCreateInfoCount=1,.pQueueCreateInfos=&qci,
        .enabledExtensionCount=1,.ppEnabledExtensionNames=enenames};
    fprintf(stderr,"[dbg] creating device fam=%u\n",fam);
    VkDevice dev; CHECK(vkCreateDevice(phys,&dci,NULL,&dev));
    fprintf(stderr,"[dbg] device ok\n");

    VkShaderModuleCreateInfo smci={.sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize=(size_t)len,.pCode=spv};
    fprintf(stderr,"[dbg] creating sm len=%ld\n",len);
    VkShaderModule sm; CHECK(vkCreateShaderModule(dev,&smci,NULL,&sm));
    fprintf(stderr,"[dbg] sm ok\n");
    free(spv);
    VkDescriptorSetLayoutBinding binds[4];
    for (uint32_t i=0;i<4;i++){ binds[i]=(VkDescriptorSetLayoutBinding){.binding=i,
        .descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.descriptorCount=1,
        .stageFlags=VK_SHADER_STAGE_COMPUTE_BIT}; }
    VkDescriptorSetLayoutCreateInfo slci={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount=4,.pBindings=binds};
    VkDescriptorSetLayout sl; CHECK(vkCreateDescriptorSetLayout(dev,&slci,NULL,&sl));
    VkPushConstantRange pcr={.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,.offset=0,.size=24};
    VkPipelineLayoutCreateInfo plci={.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount=1,.pSetLayouts=&sl,.pushConstantRangeCount=1,.pPushConstantRanges=&pcr};
    VkPipelineLayout pl; CHECK(vkCreatePipelineLayout(dev,&plci,NULL,&pl));
    fprintf(stderr,"[dbg] layout ok\n");
    VkPipelineShaderStageCreateInfo stage={.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage=VK_SHADER_STAGE_COMPUTE_BIT,.module=sm,.pName="main"};
    VkComputePipelineCreateInfo pci={.sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage=stage,.layout=pl};
    VkPipeline pipe; CHECK(vkCreateComputePipelines(dev,VK_NULL_HANDLE,1,&pci,NULL,&pipe));
    fprintf(stderr,"[dbg] pipeline ok\n");

    PFN_vkGetPipelineExecutablePropertiesKHR getProps =
        (PFN_vkGetPipelineExecutablePropertiesKHR)vkGetDeviceProcAddr(dev,"vkGetPipelineExecutablePropertiesKHR");
    PFN_vkGetPipelineExecutableStatisticsKHR getStats =
        (PFN_vkGetPipelineExecutableStatisticsKHR)vkGetDeviceProcAddr(dev,"vkGetPipelineExecutableStatisticsKHR");
    PFN_vkGetPipelineExecutableInternalRepresentationsKHR getIR =
        (PFN_vkGetPipelineExecutableInternalRepresentationsKHR)vkGetDeviceProcAddr(dev,"vkGetPipelineExecutableInternalRepresentationsKHR");
    if(!getProps||!getStats||!getIR){fprintf(stderr,"entry points missing p=%p s=%p r=%p\n",(void*)getProps,(void*)getStats,(void*)getIR);return 4;}
    fprintf(stderr,"[dbg] entry points ok\n");

    VkPipelineInfoKHR pi={.sType=VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR,.pipeline=pipe};
    uint32_t pn=0; CHECK(getProps(dev,&pi,&pn,NULL));
    VkPipelineExecutablePropertiesKHR *props=calloc(pn,sizeof(*props));
    for(uint32_t i=0;i<pn;i++) props[i].sType=VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR;
    CHECK(getProps(dev,&pi,&pn,props));
    for(uint32_t i=0;i<pn;i++){
        printf("\n=== executable %u: %s (%s) subgroup=%u ===\n",i,props[i].name,props[i].description,props[i].subgroupSize);
        VkPipelineExecutableInfoKHR ei={.sType=VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR,.pipeline=pipe,.executableIndex=i};
        uint32_t sn=0;
        if(getStats(dev,&ei,&sn,NULL)==VK_SUCCESS && sn){
            VkPipelineExecutableStatisticKHR *st=calloc(sn,sizeof(*st));
            for(uint32_t j=0;j<sn;j++) st[j].sType=VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR;
            if(getStats(dev,&ei,&sn,st)==VK_SUCCESS){
                for(uint32_t j=0;j<sn;j++){
                    printf("  stat %-24s = ",st[j].name);
                    switch(st[j].format){
                    case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR: printf("%s",st[j].value.b32?"true":"false"); break;
                    case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR: printf("%lld",(long long)st[j].value.i64); break;
                    case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR: printf("%llu",(unsigned long long)st[j].value.u64); break;
                    case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_FLOAT64_KHR: printf("%g",st[j].value.f64); break;
                    }
                    printf("   (%s)\n",st[j].description);
                }
            }
            free(st);
        }
        /* Internal representations (ACO asm) crash this RADV build; the
         * executable statistics above are the reliable resource data. */
    }
    return 0;
}
