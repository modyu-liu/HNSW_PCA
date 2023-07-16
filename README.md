# AI 数据索引实验


## 1. 开发与测试

本项目基于 C 语言，使用 CMake 工具进行构建，且不依赖平台特定的 API，可以在 Windows/MacOS/Linux 平台上开发测试。

### Windows

Windows 平台下的 Visual Studio 集成了 CMake 相关的工具，可参考官方文档：

https://learn.microsoft.com/zh-cn/cpp/build/cmake-projects-in-visual-studio

### Linux/MacOS

以 Ubuntu 为例，在开发前需要安装 CMake 和 GCC 相关编译工具链：

```bash
sudo apt install cmake build-essentials
```

MacOS 同样可以通过 homebrew 工具安装相关工具。

在当前目录执行以下指令构建、编译测试程序：

```bash
mkdir build && cd build
cmake ..
make
```

### 测试程序

编译完成后得到 `hnsw_test` 文件。通过将必要参数传入可执行文件中，以在现有数据集上测试算法效果，传入参数格式如下：

```bash
./hnsw_test base_file_path data_size query_file_path query_size groundtruth_file_path
```

`base_file_path` 指源数据文件路径，`data_size` 为源数据文件数据量，`query_file_path` 为查询文件路径，`query_size` 为查询数量，`groudtruth_file_path` 为正确查询结果集。

例如对于 SIFT SMALL 数据集，测试指令如下：

```bash
./hnsw_test ./dataset/siftsmall/siftsmall_base.fvecs 10000 ./dataset/siftsmall/siftsmall_query.fvecs 100 ./dataset/siftsmall/siftsmall_groundtruth.ivecs
```

对于 SIFT 数据集，测试指令如下：

```bash
./hnsw_test ./dataset/sift/sift_base.fvecs 1000000 ./dataset/sift/sift_query.fvecs 10000 ./dataset/sift/sift_groundtruth.ivecs
```

正确执行后将输出算法的执行时间和召回率：

```
data size: 10000
query size: 100
HNSW Context Initialied OK!
HNSW initialization cost: 0.0053 seconds
Benchmark started......
100 queries cost: 37.1073 seconds
Recall value: 1.0000
```

## 2. 开发任务

主要任务为基于 HNSW 算法实现 `src/hnsw.h` 和 `src/hnsw.c` 中的两个函数：

```C
HNSWContext *hnsw_init_context(const char *filename, size_t dim, size_t len); // load data and build graph
void hnsw_approximate_knn(HNSWContext *ctx, VecData *q, int *results, int k); // search KNN results
```

其中，`hnsw_init_context` 初始化 HNSW 算法的上下文，需要在这个函数中导入数据并初始化 HNSW 相关的数据结构。`hnsw_approximate_knn` 则在初始化后的 context 中进行近似 K 近邻查询。

我们已经在 `hnsw_init_context` 中实现了源数据的导入，另外在 `hnsw_approximate_knn` 中实现了一个简单的 KNN 算法以供参考。目前的实现仅能通过 SIFT SMALL 数据集的测试。

由于 HNSW 的实现需要例如优先队列、集合这样的数据结构辅助，你也可以引入 C++ STL 以提高你的编码效率。


## 3. 数据集下载

SIFT 数据集下载链接及其详细说明可以在以下网站中找到： http://corpus-texmex.irisa.fr/

SIFT SMALL 数据集下载链接：ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz

SIFT 数据集下载链接：ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz

建议先在 SIFT SMALL 数据集上进行开发和测试，保证算法的正确性后再在规模较大的 SIFT 数据集上进行性能测试和调优。
