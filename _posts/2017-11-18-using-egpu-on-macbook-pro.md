---
layout: post
title:  "在MacBook Pro上配置使用外置显卡"
date:   2017-11-18 13:42:05 -0400
categories: guide
mathjax: true
locale: zh-CN
---

在听说macOS High Sierra官方支持eGPU之后，便一直想买一块显卡，以弥补我用Mac 5年以来没怎么玩过大型3D游戏的遗憾，顺带炼一炼丹。趁着双十一这个借口，狠下心来买了一个eGPU盒子和一块1080Ti。

不过配置显卡的过程及其复杂，为了方便他人，同时备自己不时之需，在此记录一下。

声明：这里记载的方法是我综合网上各教程得到的，可能只适用于我自己的机型和配件，仅供参考。

<!--more-->

## 配置与环境

- **主机：**2015款MBP 15'，带AMD R9 M370X独显，以及Intel Iris Pro集显
- **系统：**macOS High Sierra 10.13.1 / Windows 10 Fall Creator's Update (10.0.10586)
- **eGPU盒：**Sonnet eGPF Breakaway Box (GPU-350W-TB3Z)
- **GPU：**EVGA GeForce GTX 1080 Ti SC2 (11G-P4-6593-KR)

之所以选择这个盒子，是因为苹果官方提供的开发者版eGPU就是基于这个盒子的。毕竟官方“认证”，用着放心一些。我买的是350W的盒子（因为便宜），只提供了一个8pin和一个6pin电源接口，因此必须使用同样使用这种接口的GPU。

另外，由于这个盒子使用Thunderbolt 3接口且只附赠TB3公对公连接线，因此还需要自行购买转接器。市面上只有TB3公对TB2母的转接器，不过因为是双向的，可以搭配TB2公对公连接线使用。

## 在Windows上配置

不得不说，Windows对各类硬件的支持还是完善得多。在Windows上配置非常简单，由于Boot Camp自带TB驱动，eGPU盒子即插即用，只要上NVIDIA官网下最新驱动安装即可。

不过这里有一个很坑的地方：MBP的两个TB2接口是不一样的。具体有什么差别我也没查到，但是我的盒子只有插在**靠近电源一侧**的接口才可以正常工作。如果插在另一个接口，虽然可以正常识别，但是GeForce Experience在驱动安装完成后仍然会提示需要安装驱动，无法使用。

### 使用内置显示器

如果使用外接显示器的话，至此已经可以正常使用了。但是我们可以通过进一步的配置，让内置显示器使用外接显卡渲染。这一部分的原理似乎是，在Mac启动时会检测是否存在独立显卡，如果存在则不会使用集成显卡。但是为了使用NVIDIA Optimus来让GPU为内置显示器渲染，则需要让集成显卡保持运行。

具体描述请参见链接[#1](#参考链接)。

#### 第一步：设置启动盘

这一步需要在macOS中完成。在System Preferences（系统偏好设置） > Startup Disk（启动盘）中，选择Windows BOOTCAMP分区作为启动盘，并重启。

如果选项中没有BOOTCAMP分区，可能是因为第三方NTFS驱动（比如我使用的Turexa NTFS）挂载了分区。以Turexa NTFS为例，在其设置页面中的Volumes页选择BOOTCAMP分区，勾选“Disable Turexa NTFS”，并在Disk Utility中卸载再挂载分区即可。

#### 第二步：创建USB引导盘

这一步是为了假装是macOS启动。你需要一个容量不超过4GB的U盘，并将其格式化为FAT格式。如果手头只有大容量U盘，可以通过以下操作来划分一块4GB的分区：

1. 以管理员权限运行`diskpart`；
2. 执行`list disk`，记下U盘对应的磁盘编号（假设为Disk 1）；
3. 执行`select disk 1`；
4. 执行`clean`，这将抹除U盘上所有的数据，并删除分区表；
5. 执行`create partition primary size=4000`，这将创建一个4GB的主分区；这一步操作后系统可能会弹出对话框询问是否需要格式化，关闭即可；
6. 执行`format fs=fat quick`，这将快速格式化分区为FAT格式；
7. 执行`assign letter = D`，这将为分区分配盘符D，以访问文件系统。

当然，更简单的办法应该是在macOS下使用Disk Utility（磁盘工具）完成上述操作。

之后，从 <https://github.com/0xbb/apple_set_os.efi/releases> 下载`apple_set_os.efi`。在U盘根目录下创建目录`/EFI/Boot`，并将下载的文件重命名为`bootx64.efi`放在目录中。

#### 第三步：执行gpu-switch

在 <https://github.com/0xbb/gpu-switch> 下载`gpu-switch`的Windows版本。它的作用是在下次启动系统时使用集成显卡。以管理员权限执行`integrated.bat`即可。

#### 第四步：通过EFI Boot引导

重新启动，在开机时按住左option键，选择EFI Boot启动。此时内置显示屏就是外接显卡渲染的啦。

为了验证这一点，可以在桌面右键菜单中打开NVIDIA Control Panel。如果右键菜单中没有这一项，或者点击后弹出“没有使用NVIDIA GPU的显示器”，则说明配置不成功。

## 在macOS上配置

现在macOS上已经有了NVIDIA的官方驱动支持。目前最新的WebDriver版本号为378.10.10.10.20.107，可以在 <http://www.nvidia.com/download/driverResults.aspx/126538/en-us> 下载。同时需要安装对应的CUDA驱动。

如果要在NVIDIA官网搜索最新版本的macOS驱动，则需要在产品系列选择“GeForce 600 Series”，操作系统选择“Show all Operating Systems”，然后选择对应的macOS系统版本。这是因为该驱动目前只为该系列显卡提供正式支持，对较新的显卡的支持还在beta阶段。

需要注意的是，安装驱动时需要开启System Integrity Protection（SIP）。具体方法是在开机进入macOS系统前按住Cmd+R进入恢复模式，打开命令行执行`csrutil enable`。同理，执行`csrutil disable`则可以关闭SIP。如果没有手动关闭过SIP的话，默认状态下SIP是开启的。

为了使用外置GPU，还需要做一些附加的配置。从 <https://egpu.io/wp-content/uploads/wpforo/attachments/3/3858-nvidia-egpu-v2-1013-1.zip> 下载NVIDIAEGPUSupport，并在**关闭SIP**的情况下安装。详细信息可以参考连接[#3](#参考链接)。

不过这时，如果在启动时连接了eGPU，则进入登录界面后会花屏。如果在启动后连接eGPU，在系统信息中的显卡信息处只能看到“NVIDIA Chip Model”，并不会显示具体型号。解决方法和Windows部分使用内置显示器的方法类似：将启动盘设为macOS分区，执行macOS下的`gpu-switch`，然后重启时从EFI Boot启动。此时可以正常进入登录界面，登录后可以使用[CUDA-Z](http://cuda-z.sourceforge.net/)检测GPU。

需要强调的一点是：目前macOS**不完全支持热拔插**。在连接eGPU后断开可能导致黑屏、重启、显示“五国语言”错误界面等。

### 使用内置显示器

至此，虽然使用了Windows部分的方法，但仍然没有让内置显示器用上eGPU。在关于本机的页面中，Built-in Display下面显示的仍然是Intel Iris Pro内置显卡。

链接[#2](#参考链接)中给出了一种方法，需要用到一个HDMI的“空插头”。因为手头没有这种插头，我没有尝试，等以后试过了再更新这一部分。

## 关于Thunderbolt 2的性能损失

在整个显卡→PCI-E→TB3→TB2→主机的数据通道中，各部分的理论最大带宽为：

- PCI-E：126Gbps
- TB3：32Gbps
- TB2：16Gbps

因此TB2成为了瓶颈。实际测试在macOS下，传输速度约为1200MB/s，也就是9.6Gbps。如果使用内置显示器的话，速度会更低。根据链接[#4](#参考链接)中的测试，在TB2连接下使用GTX 1080大概会有40%的性能损失，使用外置显示器可以将损失减小到20%。

另外，链接[#5](#参考链接)中指出性能损失可能也部分来自于`apple_set_os.efi`，并给出了一个解决方法。我没有仔细阅读，大家可以自行参考。

就游戏体验来说，即便只有60%的性能，大部分游戏也绰绰有余了。在Windows下使用1920x1600分辨率运行的NieR:Automata，在开启最高画质、关闭垂直同步时仍然较为流畅（主观感受，没有实际测过帧率）。对我来说大概够了。

另一方面，对于炼丹而言，计算耗时应该远高于传输耗时，因此瓶颈影响不大。不过这也只是我的猜想，还没有实测过。

## 参考链接

- [1] <https://egpu.io/forums/mac-setup/how-to-keep-mbps-irisiris-pro-activated-when-booting-into-windows-boot-camp/>
- [2] <https://egpu.io/how-to-egpu-accelerated-internal-display-macos/>
- [3] <https://egpu.io/forums/mac-setup/wip-nvidia-egpu-support-for-high-sierra/>
- [4] <https://egpu.io/forums/mac-setup/pcie-slot-dgpu-vs-thunderbolt-3-egpu-internal-display-test/>
- [5] <https://egpu.io/forums/mac-setup/mbp-tb3-port-underperformance-16xxmibs-instead-of-22xxmibs-under-macos-or-windowsapple_set_os-efi/>

