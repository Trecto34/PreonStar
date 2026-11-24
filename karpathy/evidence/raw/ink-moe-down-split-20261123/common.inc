static VkBuffer mkbuf(VkDeviceSize sz) {
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = sz,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    VkBuffer b; CK(vkCreateBuffer(dev, &bi, 0, &b));
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, b, &mr);
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    uint32_t t = 0; while (!((mr.memoryTypeBits >> t) & 1) || !(mp.memoryTypes[t].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) t++;
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size, .memoryTypeIndex = t};
    VkDeviceMemory m; CK(vkAllocateMemory(dev, &ai, 0, &m)); CK(vkBindBufferMemory(dev, b, m, 0));
    return b;
}
static kern mkkern(const char *path, uint32_t nb, VkBuffer *bufs, uint32_t push) {
    FILE *f = fopen(path, "rb"); if (!f) exit(3);
    fseek(f, 0, SEEK_END); long n = ftell(f); rewind(f);
    uint32_t *code = malloc(n); fread(code, 1, n, f); fclose(f);
    VkDescriptorSetLayoutBinding b[8]; VkDescriptorBufferInfo dbi[8];
    for (uint32_t i = 0; i < nb; i++) {
        b[i] = (VkDescriptorSetLayoutBinding){i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT};
        dbi[i] = (VkDescriptorBufferInfo){bufs[i], 0, VK_WHOLE_SIZE};
    }
    kern k; VkDescriptorSetLayout dsl;
    VkDescriptorSetLayoutCreateInfo dl = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = nb, .pBindings = b};
    CK(vkCreateDescriptorSetLayout(dev, &dl, 0, &dsl));
    VkPushConstantRange pr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, push};
    VkPipelineLayoutCreateInfo pl = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1, .pSetLayouts = &dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pr};
    CK(vkCreatePipelineLayout(dev, &pl, 0, &k.l));
    VkShaderModuleCreateInfo sm = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = n, .pCode = code};
    VkShaderModule mod; CK(vkCreateShaderModule(dev, &sm, 0, &mod));
    VkComputePipelineCreateInfo cp = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = mod, .pName = "main"}, .layout = k.l};
    CK(vkCreateComputePipelines(dev, 0, 1, &cp, 0, &k.p));
    VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nb};
    VkDescriptorPoolCreateInfo dpi = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &ps};
    VkDescriptorPool dp; CK(vkCreateDescriptorPool(dev, &dpi, 0, &dp));
    VkDescriptorSetAllocateInfo dai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = dp, .descriptorSetCount = 1, .pSetLayouts = &dsl};
    CK(vkAllocateDescriptorSets(dev, &dai, &k.s));
    VkWriteDescriptorSet w = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = k.s, .descriptorCount = nb, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = dbi};
    vkUpdateDescriptorSets(dev, 1, &w, 0, 0);
    return k;
}
static void begin(void) {
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    CK(vkResetCommandBuffer(cb, 0)); CK(vkBeginCommandBuffer(cb, &bi));
    vkCmdResetQueryPool(cb, qp, 0, 2 + 2 * LAYERS);
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, 0);
}
static void barrier(void) {
    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, 0, 0, 0);
}
static double end_us(double *per_disp) {
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, 1);
    CK(vkEndCommandBuffer(cb));
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb};
    CK(vkResetFences(dev, 1, &fence)); CK(vkQueueSubmit(q, 1, &si, fence));
    CK(vkWaitForFences(dev, 1, &fence, VK_TRUE, ~0ull));
    uint64_t t[2 + 2 * LAYERS];
    uint32_t nq = per_disp ? 2 + 2 * LAYERS : 2;
    CK(vkGetQueryPoolResults(dev, qp, 0, nq, sizeof t, t, 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
    if (per_disp) { double s = 0; for (int i = 0; i < LAYERS; i++) s += (t[3 + 2 * i] - t[2 + 2 * i]) * ns_per_tick / 1000.0; *per_disp = s / LAYERS; }
    return (t[1] - t[0]) * ns_per_tick / 1000.0;
}
static int cmpd(const void *a, const void *b) { double x = *(double *)a, y = *(double *)b; return (x > y) - (x < y); }
