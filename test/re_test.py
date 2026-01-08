import re

# 示例文档
text = """
## `KubeletConfiguration`

KubeletConfiguration 中包含 kubelet 的配置。

字段| 描述  
---|---  
`apiVersion`<br>string| `kubelet.config.k8s.io/v1beta1`  
`kind`<br>string| `KubeletConfiguration`  
`enableServer` **[必需]** <br>`bool`| `enableServer` 会启用 kubelet 的安全服务器。注意：kubelet 的不安全端口由 `readOnlyPort` 选项控制。默认值：`true`  
`staticPodPath`<br>`string`| `staticPodPath` 是指向要运行的本地（静态）Pod 的目录， 或者指向某个静态 Pod 文件的路径。默认值：""  
`podLogsDir`<br>`string`| podLogsDir 是 kubelet 用于放置 Pod 日志文件的自定义根目录路径。 默认值："/var/log/pods/" 注意：不建议使用临时文件夹作为日志目录，因为它可能会在许多地方引起意外行为。  
`syncFrequency`<br>`meta/v1.Duration`| `syncFrequency` 是对运行中的容器和配置进行同步的最长周期。默认值："1m"  
`fileCheckFrequency`<br>`meta/v1.Duration`| `fileCheckFrequency` 是对配置文件中新数据进行检查的时间间隔值。默认值："20s"  
`httpCheckFrequency`<br>`meta/v1.Duration`| `httpCheckFrequency` 是对 HTTP 服务器上新数据进行检查的时间间隔值。默认值："20s"  
`staticPodURL`<br>`string`| `staticPodURL` 是访问要运行的静态 Pod 的 URL 地址。默认值：""  
`staticPodURLHeader`<br>`map[string][]string`| `staticPodURLHeader`是一个由字符串组成的映射表，其中包含的 HTTP 头部信息用于访问`podURL`。默认值：nil  
`address`<br>`string`| `address` 是 kubelet 提供服务所用的 IP 地址（设置为 0.0.0.0 使用所有网络接口提供服务）。默认值："0.0.0.0"  
`port`<br>`int32`| `port` 是 kubelet 用来提供服务所使用的端口号。 这一端口号必须介于 1 到 65535 之间，包含 1 和 65535。默认值：10250  
`readOnlyPort`<br>`int32`| `readOnlyPort` 是 kubelet 用来提供服务所使用的只读端口号。 此端口上的服务不支持身份认证或鉴权。这一端口号必须介于 1 到 65535 之间， 包含 1 和 65535。将此字段设置为 0 会禁用只读服务。默认值：0（禁用）  
`tlsCertFile`<br>`string`| `tlsCertFile` 是包含 HTTPS 所需要的 x509 证书的文件 （如果有 CA 证书，会串接到服务器证书之后）。如果`tlsCertFile` 和 `tlsPrivateKeyFile` 都没有设置，则系统会为节点的公开地址生成自签名的证书和私钥， 并将其保存到 kubelet `--cert-dir` 参数所指定的目录下。默认值：""  
`tlsPrivateKeyFile`<br>`string`| `tlsPrivateKeyFile` 是一个包含与 `tlsCertFile` 证书匹配的 X509 私钥的文件。默认值：""  
`tlsCipherSuites`<br>`[]string`| `tlsCipherSuites` 是一个字符串列表，其中包含服务器所接受的加密包名称。 请注意，TLS 1.3 密码套件是不可配置的。 列表中的每个值来自于 `tls` 包中定义的常数（https://golang.org/pkg/crypto/tls/#pkg-constants）。默认值：nil  
`tlsMinVersion`<br>`string`| `tlsMinVersion` 给出所支持的最小 TLS 版本。 字段取值来自于 `tls` 包中的常数定义（https://golang.org/pkg/crypto/tls/#pkg-constants）。默认值：""  
`rotateCertificates`<br>`bool`| `rotateCertificates` 用来启用客户端证书轮换。kubelet 会调用 `certificates.k8s.io` API 来请求新的证书。需要有一个批复人批准证书签名请求。默认值：false  
`serverTLSBootstrap`<br>`bool`| `serverTLSBootstrap` 用来启用服务器证书引导。系统不再使用自签名的服务证书， kubelet 会调用 `certificates.k8s.io` API 来请求证书。 需要有一个批复人来批准证书签名请求（CSR）。 设置此字段时，`RotateKubeletServerCertificate` 特性必须被启用。默认值：false  
`authentication`<br>`KubeletAuthentication`| `authentication` 设置发送给 kubelet 服务器的请求是如何进行身份认证的。默认值：`anonymous: enabled: false webhook: enabled: true cacheTTL: "2m"`  
`authorization`<br>`KubeletAuthorization`| `authorization` 设置发送给 kubelet 服务器的请求是如何进行鉴权的。默认值：`mode: Webhook webhook: cacheAuthorizedTTL: "5m" cacheUnauthorizedTTL: "30s"`  
`registryPullQPS`<br>`int32`| `registryPullQPS` 是每秒钟可以执行的镜像仓库拉取操作限值。 此值必须不能为负数。将其设置为 0 表示没有限值。默认值：5  
`registryBurst`<br>`int32`| `registryBurst` 是突发性镜像拉取的上限值，允许镜像拉取临时上升到所指定数量， 不过仍然不超过 `registryPullQPS` 所设置的约束。此值必须是非负值。 只有 `registryPullQPS` 参数值大于 0 时才会使用此设置。默认值：10  
`imagePullCredentialsVerificationPolicy`<br>`ImagePullCredentialsVerificationPolicy`| `imagePullCredentialsVerificationPolicy` 决定当 Pod 请求节点上已存在的镜像时，凭据应被如何验证： * NeverVerify 节点上的任何人都可以使用该节点上存在的所有镜像<br> * NeverVerifyPreloadedImages 由 kubelet 以外的方式拉取到节点上的镜像可以在不重新验证凭据的情况下使用<br> * NeverVerifyAllowlistedImages 类似于 "NeverVerifyPreloadedImages"，但只有源于preloadedImagesVerificationAllowlist的节点镜像无需重新验证<br> * AlwaysVerify 所有镜像都需要重新验证凭据  
`preloadedImagesVerificationAllowlist`<br>`[]string`| `preloadedImagesVerificationAllowlist` 指定一个镜像列表，对于 `imagePullCredentialsVerificationPolicy` 设置为 "NeverVerifyAllowlistedImages" 时，这些镜像可免于重新验证凭据。 此列表支持以 "/*" 结尾的路径通配符。请仅使用不带镜像标签或摘要的镜像规约。  
`eventRecordQPS`<br>`int32`| `eventRecordQPS` 设置每秒钟可创建的事件个数上限。如果此值为 0， 则表示没有限制。此值不能设置为负数。默认值：50  
`eventBurst`<br>`int32`| `eventBurst` 是突发性事件创建的上限值，允许事件创建临时上升到所指定数量， 不过仍然不超过 `eventRecordQPS`所设置的约束。此值必须是非负值， 且只有 `eventRecordQPS` > 0 时才会使用此设置。默认值：100  
`enableDebuggingHandlers`<br>`bool`| `enableDebuggingHandlers` 启用服务器上用来访问日志、 在本地运行容器和命令的端点，包括 `exec`、`attach`、 `logs` 和 `portforward` 等功能。默认值：true  
`enableContentionProfiling`<br>`bool`| `enableContentionProfiling` 用于启用阻塞性能分析， 仅用于 `enableDebuggingHandlers` 为 `true` 的场合。默认值：false  
`healthzPort`<br>`int32`| `healthzPort` 是本地主机上提供 `healthz` 端点的端口 （设置值为 0 时表示禁止）。合法值介于 1 和 65535 之间。默认值：10248  
`healthzBindAddress`<br>`string`| `healthzBindAddress` 是 `healthz` 服务器用来提供服务的 IP 地址。``默认值："127.0.0.1"  
`oomScoreAdj`<br>`int32`| `oomScoreAdj` 是为 kubelet 进程设置的 `oom-score-adj` 值。 所设置的取值要在 [-1000, 1000] 范围之内。默认值：-999  
`clusterDomain`<br>`string`| `clusterDomain` 是集群的 DNS 域名。如果设置了此字段，kubelet 会配置所有容器，使之在搜索主机的搜索域的同时也搜索这里指定的 DNS 域。默认值：""  
`clusterDNS`<br>`[]string`| `clusterDNS` 是集群 DNS 服务器的 IP 地址的列表。 如果设置了，kubelet 将会配置所有容器使用这里的 IP 地址而不是宿主系统上的 DNS 服务器来完成 DNS 解析。默认值：nil  
`streamingConnectionIdleTimeout`<br>`meta/v1.Duration`| `streamingConnectionIdleTimeout` 设置流式连接在被自动关闭之前可以空闲的最长时间。 弃用：此字段不再有作用。 默认值："4h"  
`nodeStatusUpdateFrequency`<br>`meta/v1.Duration`| `nodeStatusUpdateFrequency` 是 kubelet 计算节点状态的频率。 如果未启用节点租约特性，这一字段设置的也是 kubelet 向控制面投递节点状态的频率。注意：如果节点租约特性未被启用，更改此参数设置时要非常小心， 所设置的参数值必须与节点控制器的 `nodeMonitorGracePeriod` 协同。默认值："10s"  
`nodeStatusReportFrequency`<br>`meta/v1.Duration`| `nodeStatusReportFrequency` 是节点状态未发生变化时，kubelet 向控制面更新节点状态的频率。如果节点状态发生变化，则 kubelet 会忽略这一频率设置， 立即更新节点状态。此字段仅当启用了节点租约特性时才被使用。`nodeStatusReportFrequency` 的默认值是"5m"。不过，如果 `nodeStatusUpdateFrequency` 被显式设置了，则 `nodeStatusReportFrequency` 的默认值会等于 `nodeStatusUpdateFrequency` 值，这是为了实现向后兼容。默认值："5m"  
`nodeLeaseDurationSeconds`<br>`int32`| `nodeLeaseDurationSeconds` 是 kubelet 会在其对应的 Lease 对象上设置的时长值。 `NodeLease` 让 kubelet 来在 `kube-node-lease` 名字空间中创建按节点名称命名的租约并定期执行续约操作，并通过这种机制来了解节点健康状况。如果租约过期，则节点可被视作不健康。根据 KEP-0009 约定，目前的租约每 10 秒钟续约一次。 在将来，租约的续约时间间隔可能会根据租约的时长来设置。此字段的取值必须大于零。默认值：40  
`imageMinimumGCAge`<br>`meta/v1.Duration`| `imageMinimumGCAge` 是对未使用镜像进行垃圾收集之前允许其存在的时长。默认值："2m"  
`imageMaximumGCAge`<br>`meta/v1.Duration`| `imageMaximumGCAge` 是对未使用镜像进行垃圾收集之前允许其存在的时长。 此字段的默认值为 "0s"，表示禁用此字段，这意味着镜像不会因为过长时间不使用而被垃圾收集。默认值："0s"（已禁用）  
`imageGCHighThresholdPercent`<br>`int32`| `imageGCHighThresholdPercent` 所给的是镜像的磁盘用量百分数， 一旦镜像用量超过此阈值，则镜像垃圾收集会一直运行。百分比是用这里的值除以 100 得到的，所以此字段取值必须介于 0 和 100 之间，包括 0 和 100。如果设置了此字段， 则取值必须大于 `imageGCLowThresholdPercent` 取值。默认值：85  
`imageGCLowThresholdPercent`<br>`int32`| `imageGCLowThresholdPercent` 所给的是镜像的磁盘用量百分数， 镜像用量低于此阈值时不会执行镜像垃圾收集操作。垃圾收集操作也将此作为最低磁盘用量边界。 百分比是用这里的值除以 100 得到的，所以此字段取值必须介于 0 和 100 之间，包括 0 和 100。 如果设置了此字段，则取值必须小于 `imageGCHighThresholdPercent` 取值。默认值：80  
`volumeStatsAggPeriod`<br>`meta/v1.Duration`| `volumeStatsAggPeriod` 是计算和缓存所有 Pod 磁盘用量的频率。默认值："1m"  
`kubeletCgroups`<br>`string`| `kubeletCgroups` 是用来隔离 kubelet 的控制组（CGroup）的绝对名称。默认值：""  
`systemCgroups`<br>`string`| `systemCgroups` 是用来放置那些未被容器化的、非内核的进程的控制组 （CGroup）的绝对名称。设置为空字符串表示没有这类容器。回滚此字段设置需要重启节点。 当此字段非空时，必须设置 `cgroupRoot` 字段。默认值：""  
`cgroupRoot`<br>`string`| `cgroupRoot` 是用来运行 Pod 的控制组（CGroup）。 容器运行时会尽可能处理此字段的设置值。  
`cgroupsPerQOS`<br>`bool`| `cgroupsPerQOS` 用来启用基于 QoS 的控制组（CGroup）层次结构： 顶层的控制组用于不同 QoS 类，所有 `Burstable` 和 `BestEffort` Pod 都会被放置到对应的顶级 QoS 控制组下。默认值：true  
`cgroupDriver`<br>`string`| `cgroupDriver` 是 kubelet 用来操控宿主系统上控制组（CGroup） 的驱动程序（cgroupfs 或 systemd）。默认值："cgroupfs"  
`cpuManagerPolicy`<br>`string`| `cpuManagerPolicy` 是要使用的策略名称。默认值："None"  
`singleProcessOOMKill`<br>`bool`| 如果 `singleProcessOOMKill` 为 true，将阻止在 cgroup v2 中为容器 cgroup 设置 `memory.oom.group` 标志。 这会导致容器中的单个进程因 OOM 被单独杀死，而不是作为一个组被杀死。 这意味着如果为 true，其行为与 cgroup v1 的行为一致。 当你未指定值时，默认值将被自动确定。 在 Windows 这类非 Linux 系统上，仅允许 null（或不设置）。 在 cgroup v1 Linux 上，仅允许 null（或不设置）和 true。 在 cgroup v2 Linux 上，允许 null（或不设置）、true 和 false。默认值为 false。  
`cpuManagerPolicyOptions`<br>`map[string]string`| `cpuManagerPolicyOptions` 是一组 `key=value` 键值映射， 容许通过额外的选项来精细调整 CPU 管理器策略的行为。默认值：nil  
`cpuManagerReconcilePeriod`<br>`meta/v1.Duration`| `cpuManagerReconcilePeriod` 是 CPU 管理器的协调周期时长。 默认值："10s"  
`memoryManagerPolicy`<br>`string`| `memoryManagerPolicy` 是内存管理器要使用的策略的名称。 要求启用 `MemoryManager` 特性门控。默认值："none"  
`topologyManagerPolicy`<br>`string`| `topologyManagerPolicy` 是要使用的拓扑管理器策略名称。合法值包括： * restricted：kubelet 仅接受在所请求资源上实现最佳 NUMA 对齐的 Pod。<br> * best-effort：kubelet 会优选在 CPU 和设备资源上实现 NUMA 对齐的 Pod。<br> * none：kubelet 不了解 Pod CPU 和设备资源 NUMA 对齐需求。<br> * single-numa-node：kubelet 仅允许在 CPU 和设备资源上对齐到同一 NUMA 节点的 Pod。默认值："none"  
`topologyManagerScope`<br>`string`| `topologyManagerScope` 代表的是拓扑提示生成的范围， 拓扑提示信息由提示提供者生成，提供给拓扑管理器。合法值包括： * container：拓扑策略是按每个容器来实施的。<br> * pod：拓扑策略是按每个 Pod 来实施的。默认值："container"  
`topologyManagerPolicyOptions`<br>`map[string]string`| TopologyManagerPolicyOptions 是一组 key=value 键值映射，容许设置额外的选项来微调拓扑管理器策略的行为。 需要同时启用 "TopologyManager" 和 "TopologyManagerPolicyOptions" 特性门控。 默认值：nil  
`qosReserved`<br>`map[string]string`| `qosReserved` 是一组从资源名称到百分比值的映射，用来为 `Guaranteed` QoS 类型的负载预留供其独占使用的资源百分比。目前支持的资源为："memory"。 需要启用 `QOSReserved` 特性门控。默认值：nil  
`runtimeRequestTimeout`<br>`meta/v1.Duration`| `runtimeRequestTimeout` 用来设置除长期运行的请求（`pull`、 `logs`、`exec` 和 `attach`）之外所有运行时请求的超时时长。默认值："2m"  
`hairpinMode`<br>`string`| `hairpinMode` 设置 kubelet 如何为发夹模式数据包配置容器网桥。 设置此字段可以让 Service 中的端点在尝试访问自身 Service 时将服务请求路由的自身。 可选值有： * "promiscuous-bridge"：将容器网桥设置为混杂模式。<br> * "hairpin-veth"：在容器的 veth 接口上设置发夹模式标记。<br> * "none"：什么也不做。一般而言，用户必须设置 `--hairpin-mode=hairpin-veth` 才能实现发夹模式的网络地址转译 （NAT），因为混杂模式的网桥要求存在一个名为 `cbr0` 的容器网桥。默认值："promiscuous-bridge"  
`maxPods`<br>`int32`| `maxPods` 是此 kubelet 上课运行的 Pod 个数上限。此值必须为非负整数。默认值：110  
`podCIDR`<br>`string`| `podCIDR` 是用来设置 Pod IP 地址的 CIDR 值，仅用于独立部署模式。 运行于集群模式时，这一数值会从控制面获得。默认值：""  
`podPidsLimit`<br>`int64`| `podPidsLimit` 是每个 Pod 中可使用的 PID 个数上限。默认值：-1  
`resolvConf`<br>`string`| `resolvConf` 是一个域名解析配置文件，用作容器 DNS 解析配置的基础。如果此值设置为空字符串，则会覆盖 DNS 解析的默认配置，本质上相当于禁用了 DNS 查询。默认值："/etc/resolv.conf"  
`runOnce`<br>`bool`| `runOnce` 字段被设置时，kubelet 会咨询 API 服务器一次并获得 Pod 列表， 运行在静态 Pod 文件中指定的 Pod 及这里所获得的 Pod，然后退出。默认值：false  
`cpuCFSQuota`<br>`bool`| `cpuCFSQuota` 允许为设置了 CPU 限制的容器实施 CPU CFS 配额约束。默认值：true  
`cpuCFSQuotaPeriod`<br>`meta/v1.Duration`| `cpuCFSQuotaPeriod` 设置 CPU CFS 配额周期值，`cpu.cfs_period_us`。 此值需要介于 1 毫秒和 1 秒之间，包含 1 毫秒和 1 秒。 此功能要求启用 `CustomCPUCFSQuotaPeriod` 特性门控被启用。默认值："100ms"  
`nodeStatusMaxImages`<br>`int32`| `nodeStatusMaxImages` 限制 `Node.status.images` 中报告的镜像数量。 此值必须大于 -2。注意：如果设置为 -1，则不会对镜像数量做限制；如果设置为 0，则不会返回任何镜像。默认值：50  
`maxOpenFiles`<br>`int64`| `maxOpenFiles` 是 kubelet 进程可以打开的文件个数。此值必须不能为负数。默认值：1000000  
`contentType`<br>`string`| `contentType` 是向 API 服务器发送请求时使用的内容类型。默认值："application/vnd.kubernetes.protobuf"  
`kubeAPIQPS`<br>`int32`| `kubeAPIQPS` 设置与 Kubernetes API 服务器通信时要使用的 QPS（每秒查询数）。默认值：50  
`kubeAPIBurst`<br>`int32`| `kubeAPIBurst` 设置与 Kubernetes API 服务器通信时突发的流量级别。 此字段取值不可以是负数。默认值：100  
`serializeImagePulls`<br>`bool`| `serializeImagePulls` 被启用时会通知 kubelet 每次仅拉取一个镜像。 我们建议 _不要_ 在所运行的 Docker 守护进程版本低于 1.9、使用 aufs 存储后端的节点上更改默认值。详细信息可参见 Issue #10959。默认值：true  
`maxParallelImagePulls`<br>`int32`| `maxParallelImagePulls` 设置并行拉取镜像的最大数量。 如果 `serializeImagePulls` 为 true，则无法设置此字段。 把它设置为 nil 意味着没有限制。默认值：nil  
`evictionHard`<br>`map[string]string`| `evictionHard` 是一个映射，是从信号名称到定义硬性驱逐阈值的映射。 例如：`{"memory.available": "300Mi"}`。 如果希望显式地禁用，可以在任意资源上将其阈值设置为 0% 或 100%。默认值：`memory.available: "100Mi"<br>nodefs.available: "10%"<br>nodefs.inodesFree: "5%"<br>imagefs.available: "15%"`  
`evictionSoft`<br>`map[string]string`| `evictionSoft` 是一个映射，是从信号名称到定义软性驱逐阈值的映射。 例如：`{"memory.available": "300Mi"}`。默认值：nil  
`evictionSoftGracePeriod`<br>`map[string]string`| `evictionSoftGracePeriod` 是一个映射，是从信号名称到每个软性驱逐信号的宽限期限。 例如：`{"memory.available": "30s"}`。默认值：nil  
`evictionPressureTransitionPeriod`<br>`meta/v1.Duration`| `evictionPressureTransitionPeriod` 设置 kubelet 离开驱逐压力状况之前必须要等待的时长。0s 的时长将被转换为默认值 5m。默认值："5m"  
`evictionMaxPodGracePeriod`<br>`int32`| `evictionMaxPodGracePeriod` 是指达到软性逐出阈值而引起 Pod 终止时， 可以赋予的宽限期限最大值（按秒计）。这个值本质上限制了软性逐出事件发生时， Pod 可以获得的 `terminationGracePeriodSeconds`。 Pod 的有效宽限期计算为： min(evictionMaxPodGracePeriod, pod.terminationGracePeriodSeconds)。 注意：负值将导致 Pod 立即被终止，就如同该值为 0 一样。 默认值：0  
`evictionMinimumReclaim`<br>`map[string]string`| `evictionMinimumReclaim` 是一个映射，定义信号名称与最小回收量数值之间的关系。 最小回收量指的是资源压力较大而执行 Pod 驱逐操作时，kubelet 对给定资源的最小回收量。 例如：`{"imagefs.available": "2Gi"}`。默认值：nil  
`mergeDefaultEvictionSettings`<br>`bool`| `mergeDefaultEvictionSettings` 表示是否应将 evictionHard、evictionSoft、 evictionSoftGracePeriod 和 evictionMinimumReclaim 字段的默认值合并到此配置中为这些字段指定的取值中。 在此配置中显式指定的信号优先生效。未在此配置中指定的信号将继承其默认值。 如果设置为 false，并且此配置中指定了任一信号，则此配置中未指定的其他信号将被设置为 0。 此字段适用于合并存在默认值的字段，目前仅 evictionHard 有默认值。 默认值：false。  
`podsPerCore`<br>`int32`| `podsPerCore` 设置的是每个核上 Pod 个数上限。此值不能超过 `maxPods`。 所设值必须是非负整数。如果设置为 0，则意味着对 Pod 个数没有限制。默认值：0  
`enableControllerAttachDetach`<br>`bool`| `enableControllerAttachDetach` 用来允许 Attach/Detach 控制器管理调度到本节点的卷的挂接（attachment）和解除挂接（detachement）， 并且禁止 kubelet 执行任何 attach/detach 操作。注意：kubelet 不支持挂接 CSI 卷和解除挂接， 因此对于该用例，此选项必须为 true。默认值：true  
`protectKernelDefaults`<br>`bool`| `protectKernelDefaults` 设置为 `true` 时，会令 kubelet 在发现内核参数与预期不符时出错退出。若此字段设置为 `false`，则 kubelet 会尝试更改内核参数以满足其预期。默认值：false  
`makeIPTablesUtilChains`<br>`bool`| `makeIPTablesUtilChains` 设置为 `true` 时，相当于允许 kubelet 在 iptables 中创建 KUBE-IPTABLES-HINT 链，提示其他组件有关系统上 iptables 的配置。默认值：true  
`iptablesMasqueradeBit`<br>`int32`| `iptablesMasqueradeBit` 以前用于控制 KUBE-MARK-MASQ 链的创建。已弃用：不再有任何效果。默认值：14  
`iptablesDropBit`<br>`int32`| `iptablesDropBit` 以前用于控制 KUBE-MARK-DROP 链的创建。已弃用：不再有任何效果。默认值：15  
`featureGates`<br>`map[string]bool`| `featureGates` 是一个从功能特性名称到布尔值的映射，用来启用或禁用实验性的功能。 此字段可逐条更改文件 "k8s.io/kubernetes/pkg/features/kube_features.go" 中所给的内置默认值。默认值：nil  
`failSwapOn`<br>`bool`| `failSwapOn` 通知 kubelet 在节点上启用交换分区时拒绝启动。默认值：true  
`memorySwap`<br>`MemorySwapConfiguration`| `memorySwap` 配置容器负载可用的交换内存。  
`containerLogMaxSize`<br>`string`| `containerLogMaxSize` 是定义容器日志文件被轮转之前可以到达的最大尺寸。 例如："5Mi" 或 "256Ki"。默认值："10Mi"  
`containerLogMaxFiles`<br>`int32`| `containerLogMaxFiles` 设置每个容器可以存在的日志文件个数上限。默认值："5"  
`containerLogMaxWorkers`<br>`int32`| `containerLogMaxWorkers` 指定执行日志轮换操作所需的并发工作程序的最大数量。 将此计数设置为 1，以禁用并发日志轮换工作流程。 默认值：1  
`containerLogMonitorInterval`<br>`meta/v1.Duration`| `containerLogMonitorInterval` 指定监视容器日志以执行日志轮转操作的持续时间。 默认为 10s，但可以根据日志生成率和需要轮换的大小定制为较小的值。 默认值：10s  
`configMapAndSecretChangeDetectionStrategy`<br>`ResourceChangeDetectionStrategy`| `configMapAndSecretChangeDetectionStrategy` 是 ConfigMap 和 Secret 管理器的运行模式。合法值包括： * Get：kubelet 从 API 服务器直接取回必要的对象；<br> * Cache：kubelet 使用 TTL 缓存来管理来自 API 服务器的对象；<br> * Watch：kubelet 使用 watch 操作来观察所关心的对象的变更。默认值："Watch"  
`systemReserved`<br>`map[string]string`| `systemReserved` 是一组`资源名称=资源数量`对， 用来描述为非 Kubernetes 组件预留的资源（例如：'cpu=200m,memory=150G'）。目前仅支持 CPU 和内存。更多细节可参见 https://kubernetes.io/zh-cn/docs/tasks/administer-cluster/reserve-compute-resources默认值：Nil  
`kubeReserved`<br>`map[string]string`| `kubeReserved` 是一组`资源名称=资源数量`对， 用来描述为 Kubernetes 系统组件预留的资源（例如：'cpu=200m,memory=150G'）。 目前支持 CPU、内存和根文件系统的本地存储。 更多细节可参见 https://kubernetes.io/zh-cn/docs/tasks/administer-cluster/reserve-compute-resources默认值：Nil  
`reservedSystemCPUs` **[必需]** <br>`string`| `reservedSystemCPUs` 选项设置为宿主级系统线程和 Kubernetes 相关线程所预留的 CPU 列表。此字段提供的是一种“静态”的 CPU 列表，而不是像 `systemReserved` 和 `kubeReserved` 所提供的“动态”列表。 此选项不支持 `systemReservedCgroup` 或 `kubeReservedCgroup`。  
`showHiddenMetricsForVersion`<br>`string`| `showHiddenMetricsForVersion` 是你希望显示隐藏度量值的上一版本。 只有上一个次版本是有意义的，其他值都是不允许的。 字段值的格式为 `<major>.<minor>`，例如：`1.16`。 此格式的目的是为了确保在下一个版本中有新的度量值被隐藏时，你有机会注意到这类变化， 而不是当这些度量值在其后的版本中彻底去除时来不及应对。``默认值：""  
`systemReservedCgroup`<br>`string`| `systemReservedCgroup` 帮助 kubelet 识别用来为 OS 系统级守护进程实施 `systemReserved` 计算资源预留时使用的顶级控制组（CGroup）。 更多细节参阅节点可分配资源。默认值：""  
`kubeReservedCgroup`<br>`string`| `kubeReservedCgroup` 帮助 kubelet 识别用来为 Kubernetes 节点系统级守护进程实施 `kubeReserved` 计算资源预留时使用的顶级控制组（CGroup）。 更多细节参阅节点可分配资源默认值：""  
`enforceNodeAllocatable`<br>`[]string`| 此标志设置 kubelet 需要执行的各类节点可分配资源策略。此字段接受一组选项列表。 可接受的选项有 `none`、`pods`、`system-reserved` 和 `kube-reserved`。如果设置了 `none`，则字段值中不可以包含其他选项。如果列表中包含 `system-reserved`，则必须设置 `systemReservedCgroup`。如果列表中包含 `kube-reserved`，则必须设置 `kubeReservedCgroup`。这个字段只有在 `cgroupsPerQOS`被设置为 `true` 才被支持。更多细节参阅节点可分配资源。默认值：["pods"]  
`allowedUnsafeSysctls`<br>`[]string`| 用逗号分隔的白名单列表，其中包含不安全的 sysctl 或 sysctl 模式（以 `*` 结尾）。不安全的 sysctl 组有 `kernel.shm*`、`kernel.msg*`、 `kernel.sem`、`fs.mqueue.*` 和 `net.*`。例如："`kernel.msg*,net.ipv4.route.min\_pmtu`"默认值：[]  
`volumePluginDir`<br>`string`| `volumePluginDir` 是用来搜索其他第三方卷插件的目录的路径。默认值："/usr/libexec/kubernetes/kubelet-plugins/volume/exec/"  
`providerID`<br>`string`| `providerID` 字段被设置时，指定的是一个外部提供者（即云驱动）实例的唯一 ID， 该提供者可用来唯一性地标识特定节点。默认值：""  
`kernelMemcgNotification`<br>`bool`| `kernelMemcgNotification` 字段如果被设置了，会告知 kubelet 集成内核的 memcg 通知机制来确定是否超出内存逐出阈值，而不是使用轮询机制来判定。默认值：false  
`logging` **[必需]** <br>`LoggingConfiguration`| `logging`设置日志机制选项。更多的详细信息可参阅 日志选项。默认值：`Format: text`  
`enableSystemLogHandler`<br>`bool`| `enableSystemLogHandler` 用来启用通过 Web 接口 host:port/logs/ 访问系统日志的能力。默认值：true  
`enableSystemLogQuery`<br>`bool`| `enableSystemLogQuery` 启用在 /logs 端点上的节点日志查询功能。 此外，还必须启用 enableSystemLogHandler 才能使此功能起作用。 启用此特性具有安全隐患。建议仅在调试需要时才启用，其他情况下应禁用。默认值：false  
`shutdownGracePeriod`<br>`meta/v1.Duration`| `shutdownGracePeriod` 设置节点关闭期间，节点自身需要延迟以及为 Pod 提供的宽限期限的总时长。默认值："0s"  
`shutdownGracePeriodCriticalPods`<br>`meta/v1.Duration`| `shutdownGracePeriodCriticalPods`设置节点关闭期间用来终止关键性 Pod 的时长。此时长要短于`shutdownGracePeriod`。 例如，如果`shutdownGracePeriod=30s`，`shutdownGracePeriodCriticalPods=10s`， 在节点关闭期间，前 20 秒钟被预留用来体面终止普通 Pod，后 10 秒钟用来终止关键 Pod。``默认值："0s"  
`shutdownGracePeriodByPodPriority`<br>`[]ShutdownGracePeriodByPodPriority`| `shutdownGracePeriodByPodPriority` 设置基于 Pod 相关的优先级类值而确定的体面关闭时间。当 kubelet 收到关闭请求的时候，kubelet 会针对节点上运行的所有 Pod 发起关闭操作，这些关闭操作会根据 Pod 的优先级确定其宽限期限， 之后 kubelet 等待所有 Pod 退出。数组中的每个表项代表的是节点关闭时 Pod 的体面终止时间；这里的 Pod 的优先级类介于列表中当前优先级类值和下一个表项的优先级类值之间。例如，要赋予关键 Pod 10 秒钟时间来关闭，赋予优先级 >=10000 Pod 20 秒钟时间来关闭， 赋予其余的 Pod 30 秒钟来关闭。shutdownGracePeriodByPodPriority: * priority: 2000000000 shutdownGracePeriodSeconds: 10<br> * priority: 10000 shutdownGracePeriodSeconds: 20<br> * priority: 0 shutdownGracePeriodSeconds: 30在退出之前，kubelet 要等待的时间上限为节点上所有优先级类的 `shutdownGracePeriodSeconds` 的最大值。 当所有 Pod 都退出或者到达其宽限期限时，kubelet 会释放关闭防护锁。 此功能要求 `GracefulNodeShutdown` 特性门控被启用。当 `shutdownGracePeriod` 或 `shutdownGracePeriodCriticalPods` 被设置时，此配置字段必须为空。默认值：nil  
`crashLoopBackOff`<br>`CrashLoopBackOffConfig`| `crashLoopBackOff` 包含修改节点级别参数的配置，用于容器重启行为。  
`reservedMemory`<br>`[]MemoryReservation`| `reservedMemory` 给出一个逗号分隔的列表，为 NUMA 节点预留内存。 此参数仅在内存管理器功能特性语境下有意义。内存管理器不会为容器负载分配预留内存。 例如，如果你的 NUMA0 节点内存为 10Gi，`reservedMemory` 设置为在 NUMA0 上预留 1Gi 内存，内存管理器会认为其上只有 9Gi 内存可供分配。 你可以设置不同数量的 NUMA 节点和内存类型。你也可以完全忽略这个字段，不过你要清楚， 所有 NUMA 节点上预留内存的总量要等于通过 节点可分配资源设置的内存量。 如果至少有一个节点可分配参数设置值非零，则你需要设置至少一个 NUMA 节点。 此外，避免如下设置： 1\. 在配置值中存在重复项，NUMA 节点和内存类型相同，但配置值不同，这是不允许的。<br> 2. 为任何内存类型设置限制值为零。<br> 3. NUMA 节点 ID 在宿主系统上不存在。/li><br> 4. 除memory和hugepages-<size>之外的内存类型。默认值：nil  
`enableProfilingHandler`<br>`bool`| `enableProfilingHandler` 启用通过 `host:port/debug/pprof/` 接口来执行性能分析。默认值：true  
`enableDebugFlagsHandler`<br>`bool`| `enableDebugFlagsHandler` 启用通过 `host:port/debug/flags/v Web` 接口上的标志设置。默认值：true  
`seccompDefault`<br>`bool`| `seccompDefault` 字段允许针对所有负载将 `RuntimeDefault` 设置为默认的 seccomp 配置。这一设置要求对应的 `SeccompDefault` 特性门控被启用。默认值：false  
`memoryThrottlingFactor`<br>`float64`| 当设置 cgroupv2 `memory.high` 以实施 `MemoryQoS` 特性时， `memoryThrottlingFactor` 用来作为内存限制或节点可分配内存的系数。减小此系数会为容器控制组设置较低的 high 限制值，从而增大回收压力；反之， 增大此系数会降低回收压力。更多细节参见 https://kep.k8s.io/2570。默认值：0.8  
`registerWithTaints`<br>`[]core/v1.Taint`| `registerWithTaints` 是一个由污点组成的数组，包含 kubelet 注册自身时要向节点对象添加的污点。只有 `registerNode` 为 `true` 时才会起作用，并且仅在节点的最初注册时起作用。默认值：nil  
`registerNode`<br>`bool`| `registerNode` 启用向 API 服务器的自动注册。默认值：true  
`tracing`<br>`TracingConfiguration`| tracing 为 OpenTelemetry 追踪客户端设置版本化的配置信息。 参阅 https://kep.k8s.io/2832 了解更多细节。  
`localStorageCapacityIsolation`<br>`bool`| localStorageCapacityIsolation 启用本地临时存储隔离特性。默认设置为 true。 此特性允许用户为容器的临时存储设置请求/限制，并以类似的方式管理 cpu 和 memory 的请求/限制。 此特性还允许为 emptyDir 卷设置 sizeLimit，如果卷所用的磁盘超过此限制将触发 Pod 驱逐。 此特性取决于准确测定根文件系统磁盘用量的能力。对于 kind rootless 这类系统， 如果不支持此能力，则 LocalStorageCapacityIsolation 特性应被禁用。 一旦禁用，用户不应该为容器的临时存储设置请求/限制，也不应该为 emptyDir 设置 sizeLimit。 默认值：true  
`containerRuntimeEndpoint` **[必需]** <br>`string`| containerRuntimeEndpoint 是容器运行时的端点。 Linux 支持 UNIX 域套接字，而 Windows 支持命名管道和 TCP 端点。 示例：'unix:///path/to/runtime.sock', 'npipe:////./pipe/runtime'。  
`imageServiceEndpoint`<br>`string`| imageServiceEndpoint 是容器镜像服务的端点。 Linux 支持 UNIX 域套接字，而 Windows 支持命名管道和 TCP 端点。 示例：'unix:///path/to/runtime.sock'、'npipe:////./pipe/runtime'。 如果未指定，则使用 containerRuntimeEndpoint 中的值。  
`failCgroupV1`<br>`bool`| `failCgroupV1` 防止 kubelet 在使用 cgroup v1 的主机上启动。 默认情况下，此选项设置为 “false”，这意味着除非此选项被显式启用， 否则 kubelet 被允许在 cgroup v1 主机上启动。 默认值：false  
`userNamespaces`<br>`UserNamespaces`| `userNamespaces` 包含用户命名空间配置。  
"""

# 正则表达式
pattern = r'(\n)(?=`[^`\n]*`[^|\n]*?\|)'
rough_chunks = re.split(pattern, text)

# 合并分隔符到后续块
combined_chunks = []
skip_next = False
for i, chunk in enumerate(rough_chunks):
    if skip_next or chunk.strip() == '':
        skip_next = False
        continue

    # 检查当前块是否是分隔符（匹配模式）
    is_separator = any(re.match(p, chunk) for p in [pattern])

    if is_separator and i + 1 < len(rough_chunks):
        # 分隔符应合并到下一块
        combined_chunks.append(chunk + rough_chunks[i + 1])
        skip_next = True
    else:
        # 普通内容块
        combined_chunks.append(chunk)

for chunk in combined_chunks:
    print(chunk)
    print("=========")
