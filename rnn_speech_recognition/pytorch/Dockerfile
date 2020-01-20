# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG DEVEL_IMG=pytorch/pytorch:1.4-cuda10.1-cudnn7-devel
ARG RELEASE_IMG=pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime


FROM ${DEVEL_IMG} as build

WORKDIR /workspace

RUN apt-get update && apt-get install -y cmake=3.10.2-1ubuntu2.18.04.1

RUN COMMIT_SHA=c6d12f9e1562833c2b4e7ad84cb22aa4ba31d18c && \
    git clone https://github.com/HawkAaron/warp-transducer deps/warp-transducer && \
    cd deps/warp-transducer && \
    git checkout $COMMIT_SHA && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make VERBOSE=1 && \
	export CUDA_HOME="/usr/local/cuda" && \
    export WARP_RNNT_PATH=`pwd` && \
    export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME && \
    export LD_LIBRARY_PATH="$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH" && \
    export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH && \
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH && \
    export CFLAGS="-I$CUDA_HOME/include $CFLAGS" && \
    cd ../pytorch_binding && \
    python3 setup.py install --user

RUN pip install --user --global-option="--cpp_ext" --global-option="--cuda_ext" --disable-pip-version-check -U https://github.com/NVIDIA/apex/archive/b5a7c5f972bc644804408dd96d7cb515df3af07b.zip

COPY requirements.txt .
RUN pip install --user --disable-pip-version-check -U -r requirements.txt


FROM ${RELEASE_IMG}

RUN apt-get update && apt-get install -y libsndfile1=1.0.28-4ubuntu0.18.04.1 && rm -rf /var/lib/apt/lists/*

ENV PATH=/root/.local/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

COPY --from=build /workspace/deps/warp-transducer/build/libwarprnnt.so /usr/local/lib
COPY --from=build /root/.local/ /root/.local/

WORKDIR /workspace/rnnt

COPY . .
