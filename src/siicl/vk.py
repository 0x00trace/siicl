"""Run translated compute programs on a GPU through Vulkan.

`Gpu().run(translation, launch)` creates the buffers and storage images that a
siicl.translate Translation names, fills them from an siicl.interp Launch,
dispatches the grid and copies the results back into the launch's surfaces,
the same way siicl.interp.run changes them. It calls the system's libvulkan
through ctypes (pacman packages vulkan-icd-loader and a GPU driver), so it
needs no Python packages.
"""

from __future__ import annotations

import ctypes as C
import struct
from ctypes import POINTER, Structure, byref, c_char_p, c_float, c_int32, c_size_t, c_uint32, c_uint64, c_void_p
from dataclasses import dataclass, field

from siicl.interp import Launch, Surface
from siicl.translate import STATUS_WORDS, Translation

API_VERSION_1_2 = (1 << 22) | (2 << 12)
VALIDATION_LAYER = b"VK_LAYER_KHRONOS_validation"
FEATURE_SHADER_INT64 = 40  # index in VkPhysicalDeviceFeatures
FEATURE_COUNT = 55
DEVICE_TYPE_DISCRETE = 2

# Structure types
ST_APPLICATION_INFO, ST_INSTANCE, ST_QUEUE, ST_DEVICE, ST_SUBMIT, ST_ALLOCATE = 0, 1, 2, 3, 4, 5
ST_FENCE, ST_BUFFER, ST_IMAGE, ST_IMAGE_VIEW, ST_SHADER_MODULE, ST_SHADER_STAGE = 8, 12, 14, 15, 16, 18
ST_COMPUTE_PIPELINE, ST_PIPELINE_LAYOUT, ST_SET_LAYOUT, ST_POOL, ST_SET_ALLOCATE = 29, 30, 32, 33, 34
ST_WRITE_SET, ST_COMMAND_POOL, ST_COMMAND_BUFFER, ST_BEGIN, ST_IMAGE_BARRIER, ST_MEMORY_BARRIER = 35, 39, 40, 42, 45, 46

FORMAT_R32_UINT, FORMAT_R32G32B32A32_UINT = 98, 107
LAYOUT_UNDEFINED, LAYOUT_GENERAL, LAYOUT_TRANSFER_SRC, LAYOUT_TRANSFER_DST = 0, 1, 6, 7
DESCRIPTOR_STORAGE_IMAGE, DESCRIPTOR_STORAGE_BUFFER = 3, 7
STAGE_COMPUTE_BIT = 0x20
BIND_POINT_COMPUTE = 1
MEMORY_DEVICE_LOCAL, MEMORY_HOST_VISIBLE, MEMORY_HOST_COHERENT = 0x1, 0x2, 0x4
BUFFER_USAGE = 0x1 | 0x2 | 0x20  # transfer source, transfer destination, storage
IMAGE_USAGE = 0x1 | 0x2 | 0x8  # transfer source, transfer destination, storage
PIPE_TOP, PIPE_COMPUTE, PIPE_TRANSFER, PIPE_HOST = 0x1, 0x800, 0x1000, 0x4000
ACCESS_SHADER_READ, ACCESS_SHADER_WRITE, ACCESS_TRANSFER_READ, ACCESS_TRANSFER_WRITE = 0x20, 0x40, 0x800, 0x1000
ACCESS_HOST_READ, ACCESS_HOST_WRITE = 0x2000, 0x4000
QUEUE_COMPUTE = 0x2
QUEUE_FAMILY_IGNORED = 0xFFFFFFFF
ASPECT_COLOR = 0x1
TIMEOUT = 2

IMAGE_KIND = {  # surface dimension -> (image type, view type)
    "1D": (0, 0),
    "1D_ARRAY": (0, 4),
    "2D": (1, 1),
    "2D_ARRAY": (1, 5),
    "3D": (2, 2),
}

Handle = c_uint64  # non-dispatchable handles


class VulkanError(RuntimeError):
    """Vulkan isn't available, or a call failed."""


def _struct(name: str, fields: list[tuple[str, object]]) -> type[Structure]:
    return type(name, (Structure,), {"_fields_": fields})


Extent3D = _struct("Extent3D", [("width", c_uint32), ("height", c_uint32), ("depth", c_uint32)])
Offset3D = _struct("Offset3D", [("x", c_int32), ("y", c_int32), ("z", c_int32)])
ApplicationInfo = _struct(
    "ApplicationInfo",
    [
        ("sType", c_int32),
        ("pNext", c_void_p),
        ("pApplicationName", c_char_p),
        ("applicationVersion", c_uint32),
        ("pEngineName", c_char_p),
        ("engineVersion", c_uint32),
        ("apiVersion", c_uint32),
    ],
)
InstanceCreateInfo = _struct(
    "InstanceCreateInfo",
    [
        ("sType", c_int32),
        ("pNext", c_void_p),
        ("flags", c_uint32),
        ("pApplicationInfo", POINTER(ApplicationInfo)),
        ("enabledLayerCount", c_uint32),
        ("ppEnabledLayerNames", POINTER(c_char_p)),
        ("enabledExtensionCount", c_uint32),
        ("ppEnabledExtensionNames", POINTER(c_char_p)),
    ],
)
DeviceQueueCreateInfo = _struct(
    "DeviceQueueCreateInfo",
    [
        ("sType", c_int32),
        ("pNext", c_void_p),
        ("flags", c_uint32),
        ("queueFamilyIndex", c_uint32),
        ("queueCount", c_uint32),
        ("pQueuePriorities", POINTER(c_float)),
    ],
)
DeviceCreateInfo = _struct(
    "DeviceCreateInfo",
    [
        ("sType", c_int32),
        ("pNext", c_void_p),
        ("flags", c_uint32),
        ("queueCreateInfoCount", c_uint32),
        ("pQueueCreateInfos", POINTER(DeviceQueueCreateInfo)),
        ("enabledLayerCount", c_uint32),
        ("ppEnabledLayerNames", POINTER(c_char_p)),
        ("enabledExtensionCount", c_uint32),
        ("ppEnabledExtensionNames", POINTER(c_char_p)),
        ("pEnabledFeatures", POINTER(c_uint32 * FEATURE_COUNT)),
    ],
)
QueueFamilyProperties = _struct(
    "QueueFamilyProperties",
    [("queueFlags", c_uint32), ("queueCount", c_uint32), ("timestampValidBits", c_uint32), ("granularity", Extent3D)],
)
MemoryRequirements = _struct(
    "MemoryRequirements", [("size", c_uint64), ("alignment", c_uint64), ("memoryTypeBits", c_uint32)]
)
MemoryType = _struct("MemoryType", [("propertyFlags", c_uint32), ("heapIndex", c_uint32)])
MemoryHeap = _struct("MemoryHeap", [("size", c_uint64), ("flags", c_uint32)])
MemoryProperties = _struct(
    "MemoryProperties",
    [
        ("memoryTypeCount", c_uint32),
        ("memoryTypes", MemoryType * 32),
        ("memoryHeapCount", c_uint32),
        ("memoryHeaps", MemoryHeap * 16),
    ],
)
BufferCreateInfo = _struct(
    "BufferCreateInfo",
    [
        ("sType", c_int32),
        ("pNext", c_void_p),
        ("flags", c_uint32),
        ("size", c_uint64),
        ("usage", c_uint32),
        ("sharingMode", c_int32),
        ("queueFamilyIndexCount", c_uint32),
        ("pQueueFamilyIndices", POINTER(c_uint32)),
    ],
)
MemoryAllocateInfo = _struct(
    "MemoryAllocateInfo",
    [("sType", c_int32), ("pNext", c_void_p), ("allocationSize", c_uint64), ("memoryTypeIndex", c_uint32)],
)
ImageCreateInfo = _struct(
    "ImageCreateInfo",
    [
        ("sType", c_int32),
        ("pNext", c_void_p),
        ("flags", c_uint32),
        ("imageType", c_int32),
        ("format", c_int32),
        ("extent", Extent3D),
        ("mipLevels", c_uint32),
        ("arrayLayers", c_uint32),
        ("samples", c_uint32),
        ("tiling", c_int32),
        ("usage", c_uint32),
        ("sharingMode", c_int32),
        ("queueFamilyIndexCount", c_uint32),
        ("pQueueFamilyIndices", POINTER(c_uint32)),
        ("initialLayout", c_int32),
    ],
)
SubresourceRange = _struct(
    "SubresourceRange",
    [
        ("aspectMask", c_uint32),
        ("baseMipLevel", c_uint32),
        ("levelCount", c_uint32),
        ("baseArrayLayer", c_uint32),
        ("layerCount", c_uint32),
    ],
)
ImageViewCreateInfo = _struct(
    "ImageViewCreateInfo",
    [
        ("sType", c_int32),
        ("pNext", c_void_p),
        ("flags", c_uint32),
        ("image", Handle),
        ("viewType", c_int32),
        ("format", c_int32),
        ("components", c_int32 * 4),
        ("subresourceRange", SubresourceRange),
    ],
)
SetLayoutBinding = _struct(
    "SetLayoutBinding",
    [
        ("binding", c_uint32),
        ("descriptorType", c_int32),
        ("descriptorCount", c_uint32),
        ("stageFlags", c_uint32),
        ("pImmutableSamplers", c_void_p),
    ],
)
SetLayoutCreateInfo = _struct(
    "SetLayoutCreateInfo",
    [
        ("sType", c_int32),
        ("pNext", c_void_p),
        ("flags", c_uint32),
        ("bindingCount", c_uint32),
        ("pBindings", POINTER(SetLayoutBinding)),
    ],
)
PipelineLayoutCreateInfo = _struct(
    "PipelineLayoutCreateInfo",
    [
        ("sType", c_int32),
        ("pNext", c_void_p),
        ("flags", c_uint32),
        ("setLayoutCount", c_uint32),
        ("pSetLayouts", POINTER(Handle)),
        ("pushConstantRangeCount", c_uint32),
        ("pPushConstantRanges", c_void_p),
    ],
)
ShaderModuleCreateInfo = _struct(
    "ShaderModuleCreateInfo",
    [("sType", c_int32), ("pNext", c_void_p), ("flags", c_uint32), ("codeSize", c_size_t), ("pCode", c_void_p)],
)
ShaderStageCreateInfo = _struct(
    "ShaderStageCreateInfo",
    [
        ("sType", c_int32),
        ("pNext", c_void_p),
        ("flags", c_uint32),
        ("stage", c_uint32),
        ("module", Handle),
        ("pName", c_char_p),
        ("pSpecializationInfo", c_void_p),
    ],
)
ComputePipelineCreateInfo = _struct(
    "ComputePipelineCreateInfo",
    [
        ("sType", c_int32),
        ("pNext", c_void_p),
        ("flags", c_uint32),
        ("stage", ShaderStageCreateInfo),
        ("layout", Handle),
        ("basePipelineHandle", Handle),
        ("basePipelineIndex", c_int32),
    ],
)
PoolSize = _struct("PoolSize", [("type", c_int32), ("descriptorCount", c_uint32)])
PoolCreateInfo = _struct(
    "PoolCreateInfo",
    [
        ("sType", c_int32),
        ("pNext", c_void_p),
        ("flags", c_uint32),
        ("maxSets", c_uint32),
        ("poolSizeCount", c_uint32),
        ("pPoolSizes", POINTER(PoolSize)),
    ],
)
SetAllocateInfo = _struct(
    "SetAllocateInfo",
    [
        ("sType", c_int32),
        ("pNext", c_void_p),
        ("descriptorPool", Handle),
        ("descriptorSetCount", c_uint32),
        ("pSetLayouts", POINTER(Handle)),
    ],
)
DescriptorBufferInfo = _struct("DescriptorBufferInfo", [("buffer", Handle), ("offset", c_uint64), ("range", c_uint64)])
DescriptorImageInfo = _struct(
    "DescriptorImageInfo", [("sampler", Handle), ("imageView", Handle), ("imageLayout", c_int32)]
)
WriteDescriptorSet = _struct(
    "WriteDescriptorSet",
    [
        ("sType", c_int32),
        ("pNext", c_void_p),
        ("dstSet", Handle),
        ("dstBinding", c_uint32),
        ("dstArrayElement", c_uint32),
        ("descriptorCount", c_uint32),
        ("descriptorType", c_int32),
        ("pImageInfo", POINTER(DescriptorImageInfo)),
        ("pBufferInfo", POINTER(DescriptorBufferInfo)),
        ("pTexelBufferView", c_void_p),
    ],
)
CommandPoolCreateInfo = _struct(
    "CommandPoolCreateInfo",
    [("sType", c_int32), ("pNext", c_void_p), ("flags", c_uint32), ("queueFamilyIndex", c_uint32)],
)
CommandBufferAllocateInfo = _struct(
    "CommandBufferAllocateInfo",
    [
        ("sType", c_int32),
        ("pNext", c_void_p),
        ("commandPool", Handle),
        ("level", c_int32),
        ("commandBufferCount", c_uint32),
    ],
)
CommandBufferBeginInfo = _struct(
    "CommandBufferBeginInfo",
    [("sType", c_int32), ("pNext", c_void_p), ("flags", c_uint32), ("pInheritanceInfo", c_void_p)],
)
ImageMemoryBarrier = _struct(
    "ImageMemoryBarrier",
    [
        ("sType", c_int32),
        ("pNext", c_void_p),
        ("srcAccessMask", c_uint32),
        ("dstAccessMask", c_uint32),
        ("oldLayout", c_int32),
        ("newLayout", c_int32),
        ("srcQueueFamilyIndex", c_uint32),
        ("dstQueueFamilyIndex", c_uint32),
        ("image", Handle),
        ("subresourceRange", SubresourceRange),
    ],
)
MemoryBarrier = _struct(
    "MemoryBarrier", [("sType", c_int32), ("pNext", c_void_p), ("srcAccessMask", c_uint32), ("dstAccessMask", c_uint32)]
)
SubresourceLayers = _struct(
    "SubresourceLayers",
    [("aspectMask", c_uint32), ("mipLevel", c_uint32), ("baseArrayLayer", c_uint32), ("layerCount", c_uint32)],
)
BufferImageCopy = _struct(
    "BufferImageCopy",
    [
        ("bufferOffset", c_uint64),
        ("bufferRowLength", c_uint32),
        ("bufferImageHeight", c_uint32),
        ("imageSubresource", SubresourceLayers),
        ("imageOffset", Offset3D),
        ("imageExtent", Extent3D),
    ],
)
SubmitInfo = _struct(
    "SubmitInfo",
    [
        ("sType", c_int32),
        ("pNext", c_void_p),
        ("waitSemaphoreCount", c_uint32),
        ("pWaitSemaphores", c_void_p),
        ("pWaitDstStageMask", c_void_p),
        ("commandBufferCount", c_uint32),
        ("pCommandBuffers", POINTER(c_void_p)),
        ("signalSemaphoreCount", c_uint32),
        ("pSignalSemaphores", c_void_p),
    ],
)
FenceCreateInfo = _struct("FenceCreateInfo", [("sType", c_int32), ("pNext", c_void_p), ("flags", c_uint32)])

P = c_void_p  # pointer to a struct, passed with byref()
_SIGNATURES = {
    "vkCreateInstance": (P, P, P),
    "vkDestroyInstance": (c_void_p, P),
    "vkEnumeratePhysicalDevices": (c_void_p, P, P),
    "vkGetPhysicalDeviceProperties": (c_void_p, P),
    "vkGetPhysicalDeviceQueueFamilyProperties": (c_void_p, P, P),
    "vkGetPhysicalDeviceMemoryProperties": (c_void_p, P),
    "vkCreateDevice": (c_void_p, P, P, P),
    "vkDestroyDevice": (c_void_p, P),
    "vkGetDeviceQueue": (c_void_p, c_uint32, c_uint32, P),
    "vkDeviceWaitIdle": (c_void_p,),
    "vkCreateBuffer": (c_void_p, P, P, P),
    "vkDestroyBuffer": (c_void_p, Handle, P),
    "vkGetBufferMemoryRequirements": (c_void_p, Handle, P),
    "vkCreateImage": (c_void_p, P, P, P),
    "vkDestroyImage": (c_void_p, Handle, P),
    "vkGetImageMemoryRequirements": (c_void_p, Handle, P),
    "vkAllocateMemory": (c_void_p, P, P, P),
    "vkFreeMemory": (c_void_p, Handle, P),
    "vkBindBufferMemory": (c_void_p, Handle, Handle, c_uint64),
    "vkBindImageMemory": (c_void_p, Handle, Handle, c_uint64),
    "vkMapMemory": (c_void_p, Handle, c_uint64, c_uint64, c_uint32, P),
    "vkUnmapMemory": (c_void_p, Handle),
    "vkCreateImageView": (c_void_p, P, P, P),
    "vkDestroyImageView": (c_void_p, Handle, P),
    "vkCreateShaderModule": (c_void_p, P, P, P),
    "vkDestroyShaderModule": (c_void_p, Handle, P),
    "vkCreateDescriptorSetLayout": (c_void_p, P, P, P),
    "vkDestroyDescriptorSetLayout": (c_void_p, Handle, P),
    "vkCreatePipelineLayout": (c_void_p, P, P, P),
    "vkDestroyPipelineLayout": (c_void_p, Handle, P),
    "vkCreateComputePipelines": (c_void_p, Handle, c_uint32, P, P, P),
    "vkDestroyPipeline": (c_void_p, Handle, P),
    "vkCreateDescriptorPool": (c_void_p, P, P, P),
    "vkDestroyDescriptorPool": (c_void_p, Handle, P),
    "vkAllocateDescriptorSets": (c_void_p, P, P),
    "vkUpdateDescriptorSets": (c_void_p, c_uint32, P, c_uint32, P),
    "vkCreateCommandPool": (c_void_p, P, P, P),
    "vkDestroyCommandPool": (c_void_p, Handle, P),
    "vkAllocateCommandBuffers": (c_void_p, P, P),
    "vkBeginCommandBuffer": (c_void_p, P),
    "vkEndCommandBuffer": (c_void_p,),
    "vkCmdPipelineBarrier": (c_void_p, c_uint32, c_uint32, c_uint32, c_uint32, P, c_uint32, P, c_uint32, P),
    "vkCmdCopyBufferToImage": (c_void_p, Handle, Handle, c_int32, c_uint32, P),
    "vkCmdCopyImageToBuffer": (c_void_p, Handle, c_int32, Handle, c_uint32, P),
    "vkCmdBindPipeline": (c_void_p, c_int32, Handle),
    "vkCmdBindDescriptorSets": (c_void_p, c_int32, Handle, c_uint32, c_uint32, P, c_uint32, P),
    "vkCmdDispatch": (c_void_p, c_uint32, c_uint32, c_uint32),
    "vkCreateFence": (c_void_p, P, P, P),
    "vkDestroyFence": (c_void_p, Handle, P),
    "vkWaitForFences": (c_void_p, c_uint32, P, c_uint32, c_uint64),
    "vkQueueSubmit": (c_void_p, c_uint32, P, Handle),
}
_VOID = {
    "vkDestroyInstance",
    "vkGetPhysicalDeviceProperties",
    "vkGetPhysicalDeviceQueueFamilyProperties",
    "vkGetPhysicalDeviceMemoryProperties",
    "vkDestroyDevice",
    "vkGetDeviceQueue",
    "vkDestroyBuffer",
    "vkGetBufferMemoryRequirements",
    "vkDestroyImage",
    "vkGetImageMemoryRequirements",
    "vkFreeMemory",
    "vkUnmapMemory",
    "vkDestroyImageView",
    "vkDestroyShaderModule",
    "vkDestroyDescriptorSetLayout",
    "vkDestroyPipelineLayout",
    "vkDestroyPipeline",
    "vkDestroyDescriptorPool",
    "vkUpdateDescriptorSets",
    "vkDestroyCommandPool",
    "vkCmdPipelineBarrier",
    "vkCmdCopyBufferToImage",
    "vkCmdCopyImageToBuffer",
    "vkCmdBindPipeline",
    "vkCmdBindDescriptorSets",
    "vkCmdDispatch",
    "vkDestroyFence",
}


class _Vk:
    """The Vulkan functions, with their argument types, from the system loader."""

    def __init__(self) -> None:
        try:
            lib = C.CDLL("libvulkan.so.1")
        except OSError as err:
            raise VulkanError(f"libvulkan.so.1 not found ({err}); install vulkan-icd-loader") from err
        for name, args in _SIGNATURES.items():
            fn = getattr(lib, name)
            fn.argtypes = list(args)
            fn.restype = None if name in _VOID else c_int32
            setattr(self, name[2:], fn)


def _check(result: int, what: str) -> None:
    if result != 0:
        raise VulkanError(f"{what} failed with VkResult {result}")


@dataclass
class RunResult:
    """What a GPU run reports besides the surfaces."""

    status: int  # bits of siicl.translate.STATUS_*
    mufu: list[tuple[int, int, int]] = field(default_factory=list)  # (address, input, result) when logged
    mufu_dropped: int = 0  # logged MUFU results that didn't fit the log


class Gpu:
    """A Vulkan device with a compute queue. Use as a context manager, or call close()."""

    def __init__(self, *, validation: bool = False) -> None:
        self.vk = vk = _Vk()
        self._alive: list = []  # objects that must outlive a call
        app = ApplicationInfo(ST_APPLICATION_INFO, None, b"siicl", 1, b"siicl", 1, API_VERSION_1_2)
        layers = (c_char_p * 1)(VALIDATION_LAYER)
        info = InstanceCreateInfo(ST_INSTANCE, None, 0, C.pointer(app), int(validation), layers, 0, None)
        self.instance = c_void_p()
        _check(vk.CreateInstance(byref(info), None, byref(self.instance)), "vkCreateInstance")
        count = c_uint32()
        vk.EnumeratePhysicalDevices(self.instance, byref(count), None)
        devices = (c_void_p * count.value)()
        vk.EnumeratePhysicalDevices(self.instance, byref(count), devices)
        if not count.value:
            raise VulkanError("no Vulkan device")
        best = None
        for dev in devices:
            props = C.create_string_buffer(4096)
            vk.GetPhysicalDeviceProperties(dev, props)
            dtype = struct.unpack_from("<I", props.raw, 16)[0]
            name = props.raw[20:276].split(b"\0")[0].decode()
            if best is None or dtype == DEVICE_TYPE_DISCRETE:
                best = (dev, name)
        self.physical, self.name = best
        nfam = c_uint32()
        vk.GetPhysicalDeviceQueueFamilyProperties(self.physical, byref(nfam), None)
        families = (QueueFamilyProperties * nfam.value)()
        vk.GetPhysicalDeviceQueueFamilyProperties(self.physical, byref(nfam), families)
        self.family = next(i for i, f in enumerate(families) if f.queueFlags & QUEUE_COMPUTE)
        self.memory = MemoryProperties()
        vk.GetPhysicalDeviceMemoryProperties(self.physical, byref(self.memory))
        priority = c_float(1.0)
        queue = DeviceQueueCreateInfo(ST_QUEUE, None, 0, self.family, 1, C.pointer(priority))
        features = (c_uint32 * FEATURE_COUNT)()
        features[FEATURE_SHADER_INT64] = 1
        dinfo = DeviceCreateInfo(ST_DEVICE, None, 0, 1, C.pointer(queue), 0, None, 0, None, C.pointer(features))
        self.device = c_void_p()
        _check(vk.CreateDevice(self.physical, byref(dinfo), None, byref(self.device)), "vkCreateDevice")
        self.queue = c_void_p()
        vk.GetDeviceQueue(self.device, self.family, 0, byref(self.queue))
        pool = CommandPoolCreateInfo(ST_COMMAND_POOL, None, 0, self.family)
        self.pool = Handle()
        _check(vk.CreateCommandPool(self.device, byref(pool), None, byref(self.pool)), "vkCreateCommandPool")

    def close(self) -> None:
        if getattr(self, "device", None):
            self.vk.DeviceWaitIdle(self.device)
            self.vk.DestroyCommandPool(self.device, self.pool, None)
            self.vk.DestroyDevice(self.device, None)
            self.device = None
        if getattr(self, "instance", None):
            self.vk.DestroyInstance(self.instance, None)
            self.instance = None

    def __enter__(self) -> Gpu:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- memory

    def _memory_type(self, bits: int, flags: int) -> int:
        for i in range(self.memory.memoryTypeCount):
            if bits >> i & 1 and self.memory.memoryTypes[i].propertyFlags & flags == flags:
                return i
        raise VulkanError(f"no memory type with flags {flags:#x}")

    def _allocate(self, reqs: MemoryRequirements, flags: int, cleanup: list) -> Handle:
        info = MemoryAllocateInfo(ST_ALLOCATE, None, reqs.size, self._memory_type(reqs.memoryTypeBits, flags))
        mem = Handle()
        _check(self.vk.AllocateMemory(self.device, byref(info), None, byref(mem)), "vkAllocateMemory")
        cleanup.append(lambda: self.vk.FreeMemory(self.device, mem, None))
        return mem

    def _buffer(self, data: bytes, cleanup: list) -> tuple[Handle, int]:
        """A host-visible buffer holding data; returns it and the address of its mapping."""
        vk, dev = self.vk, self.device
        info = BufferCreateInfo(ST_BUFFER, None, 0, len(data), BUFFER_USAGE, 0, 0, None)
        buf = Handle()
        _check(vk.CreateBuffer(dev, byref(info), None, byref(buf)), "vkCreateBuffer")
        cleanup.append(lambda: vk.DestroyBuffer(dev, buf, None))
        reqs = MemoryRequirements()
        vk.GetBufferMemoryRequirements(dev, buf, byref(reqs))
        mem = self._allocate(reqs, MEMORY_HOST_VISIBLE | MEMORY_HOST_COHERENT, cleanup)
        _check(vk.BindBufferMemory(dev, buf, mem, 0), "vkBindBufferMemory")
        ptr = c_void_p()
        _check(vk.MapMemory(dev, mem, 0, len(data), 0, byref(ptr)), "vkMapMemory")
        C.memmove(ptr, data, len(data))
        return buf, ptr.value

    def _image(self, surface: Surface, dim: str, cleanup: list) -> tuple[Handle, Handle, int]:
        vk, dev = self.vk, self.device
        image_type, view_type = IMAGE_KIND[dim]
        layers = surface.depth if dim.endswith("ARRAY") else 1
        depth = 1 if dim.endswith("ARRAY") else surface.depth
        fmt = FORMAT_R32_UINT if surface.channels == 1 else FORMAT_R32G32B32A32_UINT
        extent = Extent3D(surface.width, surface.height, depth)
        info = ImageCreateInfo(
            ST_IMAGE, None, 0, image_type, fmt, extent, 1, layers, 1, 0, IMAGE_USAGE, 0, 0, None, LAYOUT_UNDEFINED
        )
        img = Handle()
        _check(vk.CreateImage(dev, byref(info), None, byref(img)), "vkCreateImage")
        cleanup.append(lambda: vk.DestroyImage(dev, img, None))
        reqs = MemoryRequirements()
        vk.GetImageMemoryRequirements(dev, img, byref(reqs))
        mem = self._allocate(reqs, MEMORY_DEVICE_LOCAL, cleanup)
        _check(vk.BindImageMemory(dev, img, mem, 0), "vkBindImageMemory")
        rng = SubresourceRange(ASPECT_COLOR, 0, 1, 0, layers)
        vinfo = ImageViewCreateInfo(ST_IMAGE_VIEW, None, 0, img, view_type, fmt, (c_int32 * 4)(), rng)
        view = Handle()
        _check(vk.CreateImageView(dev, byref(vinfo), None, byref(view)), "vkCreateImageView")
        cleanup.append(lambda: vk.DestroyImageView(dev, view, None))
        return img, view, layers

    # -- running

    def run(self, tr: Translation, launch: Launch, *, log_capacity: int = 0, timeout: float = 120.0) -> RunResult:
        """Run a translated program on the launch's resources; the launch's surfaces get the results."""
        cleanup: list = []
        try:
            return self._run(tr, launch, log_capacity, timeout, cleanup)
        finally:
            if self.device:
                self.vk.DeviceWaitIdle(self.device)
            for f in reversed(cleanup):
                f()

    def _run(self, tr: Translation, launch: Launch, log_capacity: int, timeout: float, cleanup: list) -> RunResult:
        vk, dev = self.vk, self.device
        status_words = STATUS_WORDS + 3 * (log_capacity if tr.mufu_log else 0)
        status_buf, status_ptr = self._buffer(bytes(4 * status_words), cleanup)
        writes, keep, images = [], [], []
        layout_bindings = (SetLayoutBinding * len(tr.bindings))()
        for i, b in enumerate(tr.bindings):
            kind = DESCRIPTOR_STORAGE_IMAGE if b.kind == "surface" else DESCRIPTOR_STORAGE_BUFFER
            layout_bindings[i] = SetLayoutBinding(b.binding, kind, 1, STAGE_COMPUTE_BIT, None)
            if b.kind == "surface":
                surface = launch.surfaces.get(b.handle)
                if surface is None:
                    raise VulkanError(f"the launch has no surface for handle {b.handle}")
                if surface.channels != b.channels:
                    raise VulkanError(f"surface {b.handle} has {surface.channels} channels, the shader {b.channels}")
                img, view, layers = self._image(surface, b.dim, cleanup)
                data = struct.pack(f"<{len(surface.data)}I", *surface.data)
                staging, staging_ptr = self._buffer(data, cleanup)
                images.append((img, layers, surface, b.dim, staging, staging_ptr, len(data)))
                info = DescriptorImageInfo(0, view, LAYOUT_GENERAL)
                keep.append(info)
                writes.append(
                    WriteDescriptorSet(ST_WRITE_SET, None, 0, b.binding, 0, 1, kind, C.pointer(info), None, None)
                )
                continue
            if b.kind == "status":
                buf, size = status_buf, 4 * status_words
            else:
                if b.bank not in launch.cbufs:
                    raise VulkanError(f"the launch has no constant buffer {b.bank}")
                data = bytes(launch.cbufs[b.bank])
                data += bytes(-len(data) % 16 if data else 16)  # Vulkan needs a nonempty buffer
                buf, _ = self._buffer(data, cleanup)
                size = len(data)
            info = DescriptorBufferInfo(buf, 0, size)
            keep.append(info)
            writes.append(WriteDescriptorSet(ST_WRITE_SET, None, 0, b.binding, 0, 1, kind, None, C.pointer(info), None))

        set_info = SetLayoutCreateInfo(ST_SET_LAYOUT, None, 0, len(tr.bindings), layout_bindings)
        set_layout = Handle()
        _check(
            vk.CreateDescriptorSetLayout(dev, byref(set_info), None, byref(set_layout)), "vkCreateDescriptorSetLayout"
        )
        cleanup.append(lambda: vk.DestroyDescriptorSetLayout(dev, set_layout, None))
        pl_info = PipelineLayoutCreateInfo(ST_PIPELINE_LAYOUT, None, 0, 1, C.pointer(set_layout), 0, None)
        pipeline_layout = Handle()
        _check(vk.CreatePipelineLayout(dev, byref(pl_info), None, byref(pipeline_layout)), "vkCreatePipelineLayout")
        cleanup.append(lambda: vk.DestroyPipelineLayout(dev, pipeline_layout, None))
        code = C.create_string_buffer(tr.spirv, len(tr.spirv))
        sm_info = ShaderModuleCreateInfo(ST_SHADER_MODULE, None, 0, len(tr.spirv), C.cast(code, c_void_p))
        module = Handle()
        _check(vk.CreateShaderModule(dev, byref(sm_info), None, byref(module)), "vkCreateShaderModule")
        cleanup.append(lambda: vk.DestroyShaderModule(dev, module, None))
        stage = ShaderStageCreateInfo(ST_SHADER_STAGE, None, 0, STAGE_COMPUTE_BIT, module, b"main", None)
        cp_info = ComputePipelineCreateInfo(ST_COMPUTE_PIPELINE, None, 0, stage, pipeline_layout, 0, -1)
        pipeline = Handle()
        _check(vk.CreateComputePipelines(dev, 0, 1, byref(cp_info), None, byref(pipeline)), "vkCreateComputePipelines")
        cleanup.append(lambda: vk.DestroyPipeline(dev, pipeline, None))

        sizes = (PoolSize * 2)(PoolSize(DESCRIPTOR_STORAGE_BUFFER, 64), PoolSize(DESCRIPTOR_STORAGE_IMAGE, 64))
        pool_info = PoolCreateInfo(ST_POOL, None, 0, 1, 2, sizes)
        dpool = Handle()
        _check(vk.CreateDescriptorPool(dev, byref(pool_info), None, byref(dpool)), "vkCreateDescriptorPool")
        cleanup.append(lambda: vk.DestroyDescriptorPool(dev, dpool, None))
        alloc = SetAllocateInfo(ST_SET_ALLOCATE, None, dpool, 1, C.pointer(set_layout))
        dset = Handle()
        _check(vk.AllocateDescriptorSets(dev, byref(alloc), byref(dset)), "vkAllocateDescriptorSets")
        warr = (WriteDescriptorSet * len(writes))(*writes)
        for w in warr:
            w.dstSet = dset
        vk.UpdateDescriptorSets(dev, len(writes), warr, 0, None)

        cb_info = CommandBufferAllocateInfo(ST_COMMAND_BUFFER, None, self.pool, 0, 1)
        cb = c_void_p()
        _check(vk.AllocateCommandBuffers(dev, byref(cb_info), byref(cb)), "vkAllocateCommandBuffers")
        begin = CommandBufferBeginInfo(ST_BEGIN, None, 1, None)
        _check(vk.BeginCommandBuffer(cb, byref(begin)), "vkBeginCommandBuffer")

        def image_barrier(img, layers, src, dst, old, new, src_stage, dst_stage):
            bar = ImageMemoryBarrier(
                ST_IMAGE_BARRIER, None, src, dst, old, new, QUEUE_FAMILY_IGNORED, QUEUE_FAMILY_IGNORED, img,
                SubresourceRange(ASPECT_COLOR, 0, 1, 0, layers),
            )  # fmt: skip
            vk.CmdPipelineBarrier(cb, src_stage, dst_stage, 0, 0, None, 0, None, 1, byref(bar))

        def region(surface, dim, layers):
            depth = 1 if dim.endswith("ARRAY") else surface.depth
            sub = SubresourceLayers(ASPECT_COLOR, 0, 0, layers)
            return BufferImageCopy(0, 0, 0, sub, Offset3D(0, 0, 0), Extent3D(surface.width, surface.height, depth))

        host = MemoryBarrier(ST_MEMORY_BARRIER, None, ACCESS_HOST_WRITE, ACCESS_TRANSFER_READ | ACCESS_SHADER_READ)
        vk.CmdPipelineBarrier(cb, PIPE_HOST, PIPE_TRANSFER | PIPE_COMPUTE, 0, 1, byref(host), 0, None, 0, None)
        for img, layers, surface, dim, staging, _, _ in images:
            image_barrier(
                img, layers, 0, ACCESS_TRANSFER_WRITE, LAYOUT_UNDEFINED, LAYOUT_TRANSFER_DST, PIPE_TOP, PIPE_TRANSFER
            )
            r = region(surface, dim, layers)
            vk.CmdCopyBufferToImage(cb, staging, img, LAYOUT_TRANSFER_DST, 1, byref(r))
            image_barrier(
                img, layers, ACCESS_TRANSFER_WRITE, ACCESS_SHADER_READ | ACCESS_SHADER_WRITE,
                LAYOUT_TRANSFER_DST, LAYOUT_GENERAL, PIPE_TRANSFER, PIPE_COMPUTE,
            )  # fmt: skip
        vk.CmdBindPipeline(cb, BIND_POINT_COMPUTE, pipeline)
        vk.CmdBindDescriptorSets(cb, BIND_POINT_COMPUTE, pipeline_layout, 0, 1, byref(dset), 0, None)
        vk.CmdDispatch(cb, *launch.grid)
        for img, layers, surface, dim, staging, _, _ in images:
            image_barrier(
                img, layers, ACCESS_SHADER_WRITE, ACCESS_TRANSFER_READ,
                LAYOUT_GENERAL, LAYOUT_TRANSFER_SRC, PIPE_COMPUTE, PIPE_TRANSFER,
            )  # fmt: skip
            r = region(surface, dim, layers)
            vk.CmdCopyImageToBuffer(cb, img, LAYOUT_TRANSFER_SRC, staging, 1, byref(r))
        done = MemoryBarrier(ST_MEMORY_BARRIER, None, ACCESS_SHADER_WRITE | ACCESS_TRANSFER_WRITE, ACCESS_HOST_READ)
        vk.CmdPipelineBarrier(cb, PIPE_COMPUTE | PIPE_TRANSFER, PIPE_HOST, 0, 1, byref(done), 0, None, 0, None)
        _check(vk.EndCommandBuffer(cb), "vkEndCommandBuffer")

        fence_info = FenceCreateInfo(ST_FENCE, None, 0)
        fence = Handle()
        _check(vk.CreateFence(dev, byref(fence_info), None, byref(fence)), "vkCreateFence")
        cleanup.append(lambda: vk.DestroyFence(dev, fence, None))
        cbs = (c_void_p * 1)(cb)
        submit = SubmitInfo(ST_SUBMIT, None, 0, None, None, 1, cbs, 0, None)
        _check(vk.QueueSubmit(self.queue, 1, byref(submit), fence), "vkQueueSubmit")
        result = vk.WaitForFences(dev, 1, byref(fence), 1, int(timeout * 1e9))
        if result == TIMEOUT:
            raise VulkanError(f"the GPU did not finish within {timeout:.0f} s")
        _check(result, "vkWaitForFences")

        for _, _, surface, _, _, ptr, size in images:
            surface.data = list(struct.unpack(f"<{size // 4}I", C.string_at(ptr, size)))
        words = struct.unpack(f"<{status_words}I", C.string_at(status_ptr, 4 * status_words))
        logged = words[1]
        kept = min(logged, log_capacity) if tr.mufu_log else 0
        entries = [tuple(words[STATUS_WORDS + 3 * k : STATUS_WORDS + 3 * k + 3]) for k in range(kept)]
        return RunResult(words[0], entries, logged - kept)
