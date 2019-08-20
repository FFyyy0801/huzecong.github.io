---
layout:         post
title:          "在MacBook Pro上配置使用外置显卡"
date:           2017-11-18 13:42:05 -0400
categories:     tech
locale:         zh-Hans
mathjax:        true
footnote_title: "参考链接"
toc:            true
---

在听说 macOS High Sierra 官方支持 eGPU 之后，便一直想买一块显卡，以弥补我用 Mac 5年以来没怎么玩过大型 3D 游戏的遗憾，顺带炼一炼丹。趁着双十一这个借口，狠下心来买了一个 eGPU 盒子和一块 1080 Ti。

不过配置显卡的过程及其复杂，为了方便他人，同时备自己不时之需，在此记录一下。

声明：这里记载的方法是我综合网上各教程得到的，可能只适用于我自己的机型和配件，仅供参考。

<!--more-->

## 配置与环境

- **主机：**2015款 MBP 15'，带 AMD R9 M370X 独显，以及 Intel Iris Pro 集显
- **系统：**macOS High Sierra 10.13.1 / Windows 10 Fall Creator's Update (10.0.10586)
- **eGPU盒：**Sonnet eGPF Breakaway Box (GPU-350W-TB3Z)
- **GPU：**EVGA GeForce GTX 1080 Ti SC2 (11G-P4-6593-KR)

之所以选择这个盒子，是因为苹果官方提供的开发者版 eGPU 就是基于这个盒子的。毕竟官方“认证”，用着放心一些。我买的是 350W 的盒子（因为便宜），只提供了一个 8pin 和一个 6pin 电源接口，因此必须使用同样使用这种接口的 GPU。

另外，由于这个盒子使用 Thunderbolt 3 接口且只附赠 TB3 公对公连接线，因此还需要自行购买转接器。市面上只有 TB3 公对 TB2 母的转接器，不过因为是双向的，可以搭配 TB2 公对公连接线使用。

## 在 Windows 上配置

不得不说，Windows 对各类硬件的支持还是完善得多。在 Windows 上配置非常简单，由于 Boot Camp 自带 TB 驱动，eGPU 盒子即插即用，只要上 NVIDIA 官网下最新驱动安装即可。

不过这里有一个很坑的地方：MBP 的两个 TB2 接口是不一样的。具体有什么差别我也没查到，但是我的盒子只有插在**靠近电源一侧**的接口才可以正常工作。如果插在另一个接口，虽然可以正常识别，但是 GeForce Experience 在驱动安装完成后仍然会提示需要安装驱动，无法使用。

### 使用内置显示器

如果使用外接显示器的话，至此已经可以正常使用了。但是我们可以通过进一步的配置，让内置显示器使用外接显卡渲染。这一部分的原理似乎是，在 Mac 启动时会检测是否存在独立显卡，如果存在则不会使用集成显卡。但是为了使用 NVIDIA Optimus 来让 GPU 为内置显示器渲染，则需要让集成显卡保持运行。

具体描述请参见链接[^1]。

#### 第一步：设置启动盘

这一步需要在 macOS 中完成。在 System Preferences（系统偏好设置） > Startup Disk（启动盘）中，选择 Windows BOOTCAMP 分区作为启动盘，并重启。

如果选项中没有 BOOTCAMP 分区，可能是因为第三方 NTFS 驱动（比如我使用的 Turexa NTFS）挂载了分区。以 Turexa NTFS 为例，在其设置页面中的 Volumes 页选择 BOOTCAMP 分区，勾选“Disable Turexa NTFS”，并在 Disk Utility 中卸载再挂载分区即可。

#### 第二步：创建 USB 引导盘

这一步是为了假装是 macOS 启动。你需要一个容量不超过 4GB 的U盘，并将其格式化为 FAT 格式。如果手头只有大容量U盘，可以通过以下操作来划分一块 4GB 的分区：

1. 以管理员权限运行 `diskpart`；
2. 执行 `list disk`，记下U盘对应的磁盘编号（假设为 Disk 1）；
3. 执行 `select disk 1`；
4. 执行 `clean`，这将抹除U盘上所有的数据，并删除分区表；
5. 执行 `create partition primary size=4000`，这将创建一个 4GB 的主分区；这一步操作后系统可能会弹出对话框询问是否需要格式化，关闭即可；
6. 执行 `format fs=fat quick`，这将快速格式化分区为 FAT 格式；
7. 执行 `assign letter = D`，这将为分区分配盘符 D ，以访问文件系统。

当然，更简单的办法应该是在 macOS 下使用 Disk Utility（磁盘工具）完成上述操作。

之后，下载 [`apple_set_os.efi`](https://github.com/0xbb/apple_set_os.efi/releases)。在U盘根目录下创建目录 `/EFI/Boot`，并将下载的文件重命名为 `bootx64.efi` 放在目录中。

#### 第三步：执行 `gpu-switch`

下载 [`gpu-switch`](https://github.com/0xbb/gpu-switch) 的 Windows 版本。它的作用是在下次启动系统时使用集成显卡。以管理员权限执行 `integrated.bat` 即可。

#### 第四步：通过 EFI Boot 引导

重新启动，在开机时按住左 option 键，选择 EFI Boot 启动。此时内置显示屏就是外接显卡渲染的啦。

为了验证这一点，可以在桌面右键菜单中打开 NVIDIA Control Panel。如果右键菜单中没有这一项，或者点击后弹出“没有使用 NVIDIA GPU 的显示器”，则说明配置不成功。

## 在 macOS 上配置

现在 macOS 上已经有了 NVIDIA 的官方驱动支持。目前最新的 WebDriver 版本号为 378.10.10.10.20.107，可以在 [NVIDIA 官方网站](http://www.nvidia.com/download/driverResults.aspx/126538/en-us)上下载。同时需要安装对应的 CUDA 驱动。

如果要在 NVIDIA 官网搜索最新版本的 macOS 驱动，则需要在产品系列选择“GeForce 600 Series”，操作系统选择“Show all Operating Systems”，然后选择对应的 macOS 系统版本。这是因为该驱动目前只为该系列显卡提供正式支持，对较新的显卡的支持还在 beta 阶段。

需要注意的是，安装驱动时需要开启 System Integrity Protection（SIP）。具体方法是在开机进入 macOS 系统前按住 Cmd+R 进入恢复模式，打开命令行执行 `csrutil enable`。同理，执行 `csrutil disable`则可以关闭 SIP。如果没有手动关闭过 SIP的话，默认状态下 SIP是开启的。

为了使用外置 GPU，还需要做一些附加的配置。下载 [NVIDIAEGPUSupport](https://egpu.io/wp-content/uploads/wpforo/attachments/3/3858-nvidia-egpu-v2-1013-1.zip)，并在**关闭 SIP** 的情况下安装。详细信息可以参考连接[^3]。

不过这时，如果在启动时连接了 eGPU，则进入登录界面后会花屏。如果在启动后连接 eGPU，在系统信息中的显卡信息处只能看到“NVIDIA Chip Model”，并不会显示具体型号。解决方法和 Windows 部分使用内置显示器的方法类似：将启动盘设为 macOS 分区，执行 macOS 下的 `gpu-switch`，然后重启时从 EFI Boot 启动。此时可以正常进入登录界面，登录后可以使用 [CUDA-Z](http://cuda-z.sourceforge.net/) 检测 GPU。

需要强调的一点是：目前 macOS **不完全支持热拔插**。在连接 eGPU 后断开可能导致黑屏、重启、显示“五国语言”错误界面等。

### 使用内置显示器

至此，虽然使用了 Windows 部分的方法，但仍然没有让内置显示器用上 eGPU。在关于本机的页面中，Built-in Display 下面显示的仍然是 Intel Iris Pro 内置显卡。

链接[^2]中给出了一种方法，需要用到一个 HDMI 的“空插头”。因为手头没有这种插头，我没有尝试，等以后试过了再更新这一部分。

## 关于 Thunderbolt 2 的性能损失

在整个显卡→PCI-E→TB3→TB2→主机的数据通道中，各部分的理论最大带宽为：

- PCI-E：126Gbps
- TB3：32Gbps
- TB2：16Gbps

因此 TB2 成为了瓶颈。实际测试在 macOS 下，传输速度约为 1200MB/s，也就是 9.6Gbps。如果使用内置显示器的话，速度会更低。根据链接[^4]中的测试，在 TB2 连接下使用 GTX 1080 大概会有 40% 的性能损失，使用外置显示器可以将损失减小到 20%。

另外，链接[^5]中指出性能损失可能也部分来自于 `apple_set_os.efi`，并给出了一个解决方法。我没有仔细阅读，大家可以自行参考。

就游戏体验来说，即便只有60%的性能，大部分游戏也绰绰有余了。在 Windows 下使用 1920x1600 分辨率运行的 NieR:Automata，在开启最高画质、关闭垂直同步时仍然较为流畅（主观感受，没有实际测过帧率）。对我来说大概够了。

另一方面，对于炼丹而言，计算耗时应该远高于传输耗时，因此瓶颈影响不大。不过这也只是我的猜想，还没有实测过。

[^1]: <https://egpu.io/forums/mac-setup/how-to-keep-mbps-irisiris-pro-activated-when-booting-into-windows-boot-camp/>
[^2]: <https://egpu.io/how-to-egpu-accelerated-internal-display-macos/>
[^3]: <https://egpu.io/forums/mac-setup/wip-nvidia-egpu-support-for-high-sierra/>
[^4]: <https://egpu.io/forums/mac-setup/pcie-slot-dgpu-vs-thunderbolt-3-egpu-internal-display-test/>
[^5]: <https://egpu.io/forums/mac-setup/mbp-tb3-port-underperformance-16xxmibs-instead-of-22xxmibs-under-macos-or-windowsapple_set_os-efi/>
