#include <vulkan/vulkan.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(call) do { VkResult r_ = (call); if (r_ != VK_SUCCESS) { \
    fprintf(stderr, "%s failed: %d\n", #call, r_); exit(1); } } while (0)

typedef struct { VkBuffer buffer; VkDeviceMemory memory; void *map; VkDeviceSize size; } Buf;
static VkDevice device;
static VkPhysicalDeviceMemoryProperties memprops;

static Buf make_buf(VkDeviceSize size) {
    Buf b = {.size = size};
    VkBufferCreateInfo ci = {.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size=size, .usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .sharingMode=VK_SHARING_MODE_EXCLUSIVE};
    CHECK(vkCreateBuffer(device, &ci, NULL, &b.buffer));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(device, b.buffer, &req);
    uint32_t type = UINT32_MAX;
    for (uint32_t i=0; i<memprops.memoryTypeCount; i++) {
        VkMemoryPropertyFlags flags = memprops.memoryTypes[i].propertyFlags;
        if ((req.memoryTypeBits & (1u<<i)) &&
            (flags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
            (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            type=i; break;
        }
    }
    if (type == UINT32_MAX) { fprintf(stderr,"no coherent host-visible Vulkan heap\n"); exit(1); }
    VkMemoryAllocateInfo ai = {.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize=req.size, .memoryTypeIndex=type};
    CHECK(vkAllocateMemory(device, &ai, NULL, &b.memory));
    CHECK(vkBindBufferMemory(device, b.buffer, b.memory, 0));
    CHECK(vkMapMemory(device, b.memory, 0, size, 0, &b.map));
    return b;
}

static void free_buf(Buf *b) {
    vkUnmapMemory(device,b->memory);
    vkDestroyBuffer(device,b->buffer,NULL);
    vkFreeMemory(device,b->memory,NULL);
}

static int cmp_d(const void *a, const void *b) {
    double x=*(const double*)a, y=*(const double*)b;
    return (x>y)-(x<y);
}

int main(int argc, char **argv) {
    if (argc != 2) { fprintf(stderr,"usage: %s read_copy.spv\n",argv[0]); return 2; }
    FILE *f=fopen(argv[1],"rb");
    if (!f) { perror(argv[1]); return 1; }
    fseek(f,0,SEEK_END); long spv_len=ftell(f); rewind(f);
    uint32_t *spv=malloc((size_t)spv_len);
    if (!spv || fread(spv,1,(size_t)spv_len,f)!=(size_t)spv_len) return 1;
    fclose(f);
    VkApplicationInfo app={.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName="bc250-read-copy",.apiVersion=VK_API_VERSION_1_2};
    VkInstanceCreateInfo ici={.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,.pApplicationInfo=&app};
    VkInstance instance;
    CHECK(vkCreateInstance(&ici,NULL,&instance));
    uint32_t count=0;
    CHECK(vkEnumeratePhysicalDevices(instance,&count,NULL));
    if (!count) return 1;
    VkPhysicalDevice *devices=calloc(count,sizeof(*devices));
    CHECK(vkEnumeratePhysicalDevices(instance,&count,devices));
    VkPhysicalDevice physical=devices[0]; free(devices);
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physical,&props);
    vkGetPhysicalDeviceMemoryProperties(physical,&memprops);
    uint32_t families=0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical,&families,NULL);
    VkQueueFamilyProperties *qp=calloc(families,sizeof(*qp));
    vkGetPhysicalDeviceQueueFamilyProperties(physical,&families,qp);
    uint32_t family=UINT32_MAX;
    for(uint32_t i=0;i<families;i++) if(qp[i].queueFlags&VK_QUEUE_COMPUTE_BIT) { family=i;break; }
    if(family==UINT32_MAX || !qp[family].timestampValidBits) return 1;
    free(qp);
    float priority=1.0f;
    VkDeviceQueueCreateInfo qci={.sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex=family,.queueCount=1,.pQueuePriorities=&priority};
    VkDeviceCreateInfo dci={.sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount=1,.pQueueCreateInfos=&qci};
    CHECK(vkCreateDevice(physical,&dci,NULL,&device));
    VkQueue queue;
    vkGetDeviceQueue(device,family,0,&queue);
    const uint32_t groups=4096, lanes=256;
    const VkDeviceSize bytes=512ull*1024ull*1024ull;
    Buf src=make_buf(bytes),dst=make_buf(bytes),sum=make_buf(groups*4ull);
    uint32_t *p=(uint32_t*)src.map;
    for (uint64_t i=0;i<bytes/4;i++) p[i]=(uint32_t)(i*1664525u+1013904223u);
    memset(dst.map,0,(size_t)bytes); memset(sum.map,0,groups*4u);
    VkDescriptorSetLayoutBinding binding[3]={0};
    for(uint32_t i=0;i<3;i++) {
        binding[i].binding=i; binding[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binding[i].descriptorCount=1; binding[i].stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo slci={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount=3,.pBindings=binding};
    VkDescriptorSetLayout sl;
    CHECK(vkCreateDescriptorSetLayout(device,&slci,NULL,&sl));
    VkPushConstantRange range={.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,.offset=0,.size=8};
    VkPipelineLayoutCreateInfo plci={.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount=1,.pSetLayouts=&sl,.pushConstantRangeCount=1,.pPushConstantRanges=&range};
    VkPipelineLayout pl;
    CHECK(vkCreatePipelineLayout(device,&plci,NULL,&pl));
    VkShaderModuleCreateInfo smci={.sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize=(size_t)spv_len,.pCode=spv};
    VkShaderModule sm;
    CHECK(vkCreateShaderModule(device,&smci,NULL,&sm)); free(spv);
    VkPipelineShaderStageCreateInfo stage={.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage=VK_SHADER_STAGE_COMPUTE_BIT,.module=sm,.pName="main"};
    VkComputePipelineCreateInfo pci={.sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage=stage,.layout=pl};
    VkPipeline pipeline;
    CHECK(vkCreateComputePipelines(device,VK_NULL_HANDLE,1,&pci,NULL,&pipeline));
    VkDescriptorPoolSize dps={.type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.descriptorCount=3};
    VkDescriptorPoolCreateInfo dpci={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets=1,.poolSizeCount=1,.pPoolSizes=&dps};
    VkDescriptorPool pool;
    CHECK(vkCreateDescriptorPool(device,&dpci,NULL,&pool));
    VkDescriptorSetAllocateInfo dsai={.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool=pool,.descriptorSetCount=1,.pSetLayouts=&sl};
    VkDescriptorSet set;
    CHECK(vkAllocateDescriptorSets(device,&dsai,&set));
    Buf *buffers[3]={&src,&dst,&sum};
    VkDescriptorBufferInfo infos[3]; VkWriteDescriptorSet writes[3];
    for(uint32_t i=0;i<3;i++) {
        infos[i]=(VkDescriptorBufferInfo){.buffer=buffers[i]->buffer,.offset=0,.range=buffers[i]->size};
        writes[i]=(VkWriteDescriptorSet){.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet=set,.dstBinding=i,.descriptorCount=1,
            .descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,.pBufferInfo=&infos[i]};
    }
    vkUpdateDescriptorSets(device,3,writes,0,NULL);
    VkCommandPoolCreateInfo cpci={.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,.queueFamilyIndex=family};
    VkCommandPool cmdpool;
    CHECK(vkCreateCommandPool(device,&cpci,NULL,&cmdpool));
    VkCommandBufferAllocateInfo cbai={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool=cmdpool,.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,.commandBufferCount=1};
    VkCommandBuffer cmd;
    CHECK(vkAllocateCommandBuffers(device,&cbai,&cmd));
    VkQueryPoolCreateInfo qpci={.sType=VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
        .queryType=VK_QUERY_TYPE_TIMESTAMP,.queryCount=2};
    VkQueryPool query;
    CHECK(vkCreateQueryPool(device,&qpci,NULL,&query));
    VkFenceCreateInfo fci={.sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence;
    CHECK(vkCreateFence(device,&fci,NULL,&fence));
    printf("device=%s subgroup_default_query=see vulkaninfo queue_family=%u timestamp_ns=%.3f physical_bytes=%llu\n",
        props.deviceName,family,props.limits.timestampPeriod,(unsigned long long)bytes);
    for(uint32_t mode=0;mode<3;mode++) {
        uint32_t wpt= mode==2 ? 112u : 128u;
        uint64_t logical=(uint64_t)groups*lanes*wpt;
        double ms[7];
        for(int rep=-1;rep<7;rep++) {
            CHECK(vkResetCommandBuffer(cmd,0));
            VkCommandBufferBeginInfo bi={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            CHECK(vkBeginCommandBuffer(cmd,&bi));
            vkCmdResetQueryPool(cmd,query,0,2);
            vkCmdWriteTimestamp(cmd,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,query,0);
            vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,pipeline);
            vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,pl,0,1,&set,0,NULL);
            uint32_t push[2]={wpt,mode};
            vkCmdPushConstants(cmd,pl,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(push),push);
            vkCmdDispatch(cmd,groups,1,1);
            vkCmdWriteTimestamp(cmd,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,query,1);
            VkMemoryBarrier mb={.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT,.dstAccessMask=VK_ACCESS_HOST_READ_BIT};
            vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_HOST_BIT,0,1,&mb,0,NULL,0,NULL);
            CHECK(vkEndCommandBuffer(cmd));
            CHECK(vkResetFences(device,1,&fence));
            VkSubmitInfo si={.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=1,.pCommandBuffers=&cmd};
            CHECK(vkQueueSubmit(queue,1,&si,fence));
            CHECK(vkWaitForFences(device,1,&fence,VK_TRUE,UINT64_MAX));
            uint64_t ticks[2];
            CHECK(vkGetQueryPoolResults(device,query,0,2,sizeof(ticks),ticks,sizeof(uint64_t),
                VK_QUERY_RESULT_64_BIT|VK_QUERY_RESULT_WAIT_BIT));
            if(rep>=0) ms[rep]=(double)(ticks[1]-ticks[0])*props.limits.timestampPeriod/1e6;
        }
        qsort(ms,7,sizeof(double),cmp_d);
        uint32_t *s=(uint32_t*)sum.map;
        if (s[0]==0 && s[groups-1]==0) { fprintf(stderr,"checksum absent mode=%u\n",mode); return 1; }
        if (mode==1 && (((uint32_t*)dst.map)[0]!=p[0] ||
                       ((uint32_t*)dst.map)[bytes/4-1]!=p[bytes/4-1])) {
            fprintf(stderr,"copy validation failed\n");return 1;
        }
        double gb=(double)logical*4.0/1e9;
        double traffic=mode==1 ? 2.0*gb : gb;
        printf("mode=%u useful_read_GB=%.3f copy_read_plus_write_GB=%.3f median_ms=%.3f useful_GBps=%.1f checksum=%08x/%08x samples_ms=",
            mode,gb,mode==1?traffic:0.0,ms[3],traffic/(ms[3]/1e3),s[0],s[groups-1]);
        for(int i=0;i<7;i++) printf("%s%.3f",i?",":"",ms[i]);
        putchar('\n'); fflush(stdout);
    }
    CHECK(vkDeviceWaitIdle(device));
    vkDestroyFence(device,fence,NULL); vkDestroyQueryPool(device,query,NULL);
    vkDestroyCommandPool(device,cmdpool,NULL); vkDestroyDescriptorPool(device,pool,NULL);
    vkDestroyPipeline(device,pipeline,NULL); vkDestroyShaderModule(device,sm,NULL);
    vkDestroyPipelineLayout(device,pl,NULL); vkDestroyDescriptorSetLayout(device,sl,NULL);
    free_buf(&sum);free_buf(&dst);free_buf(&src);
    vkDestroyDevice(device,NULL); vkDestroyInstance(instance,NULL);
    return 0;
}
