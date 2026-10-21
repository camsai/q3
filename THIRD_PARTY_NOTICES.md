# Third-party notices

Q3's own source code is licensed under the Apache License 2.0 ([LICENSE.md](LICENSE.md)).
Its binaries and container images also contain or load the components below, each under
its own license.

## The container images

> This image is a combined work including FFTW and is distributed under GPL-3.0-or-later
> as a whole; Q3's own source is Apache-2.0.

FFTW is GPL-2.0-or-later and every published image (`ghcr.io/<owner>/q3:<tag>`, `-mpi`,
`-cuda`) links it. Apache-2.0 is compatible with version 3 of the GPL, so the
combination is distributed under GPL-3.0-or-later. Each component keeps its own license
for its own code. Nothing here restricts the rights the GPL gives you.

In each image, `/usr/share/licenses/q3/` holds this file, Q3's `LICENSE.md` and the
license file of every dependency compiled into the binaries; the `-mpi` image adds
FFTW's and MPICH's own notices. Ubuntu installs the copyright file of every package
under `/usr/share/doc/<package>/copyright` and the GPL, LGPL and MPL texts under
`/usr/share/common-licenses/`. The `-cuda` image carries NVIDIA's license at
`/NGC-DL-CONTAINER-LICENSE`.

## Corresponding source

Every published image's release carries `q3-<tag>-container-sources.tar.gz` as an asset,
in the repository that published the image:

    https://github.com/<owner>/q3/releases/download/<tag>/q3-<tag>-container-sources.tar.gz

It holds FFTW's source exactly as the images use it (Ubuntu's source package at the
version the plain and `-cuda` images pin, and the release tarball the `-mpi` image
compiles), the Dockerfiles that build the images, and the commit of Q3 they were built
from, whose source is this repository at that commit. Each image names its bundle's URL
and SHA-256 at the end of its copy of this file and in the labels `q3.source-bundle.url`
and `q3.source-bundle.sha256`. The bundle stays available for as long as the image is
distributed.

The commit the bundle names is permanent: the published repository's history is not
rewritten after the public release, so the commit URL in the bundle's `README.txt` resolves
for as long as the image is distributed. A snapshot of Q3's source at that commit is also
available on request from the address in `LICENSE.md`'s copyright notice.

The bundle covers FFTW, the GPL component Q3 itself links. The rest of each image is
Ubuntu 24.04's binary packages, unmodified: the GNU C Library, the GCC runtime libraries
and the base image's GPL and LGPL tools. Their corresponding source is Ubuntu's published
source package for each version the image installs, at
<https://archive.ubuntu.com/ubuntu/pool/> and on Launchpad, and
`/usr/share/doc/<package>/copyright` in the image names each package's license.

## Components

| Component                                                          | License                                 | Upstream                                |
| ------------------------------------------------------------------ | --------------------------------------- | --------------------------------------- |
| FFTW 3.3.10                                                        | GPL-2.0-or-later                        | <https://www.fftw.org/>                 |
| LibXC                                                              | MPL-2.0                                 | <https://libxc.gitlab.io/>              |
| OpenBLAS, including its LAPACK                                     | BSD-3-Clause                            | <https://www.openblas.net/>             |
| spglib                                                             | BSD-3-Clause                            | <https://github.com/spglib/spglib>      |
| spdlog, with the fmt library it bundles                            | MIT                                     | <https://github.com/gabime/spdlog>      |
| yaml-cpp                                                           | MIT                                     | <https://github.com/jbeder/yaml-cpp>    |
| nlohmann/json                                                      | MIT                                     | <https://github.com/nlohmann/json>      |
| PocketFFT                                                          | BSD-3-Clause                            | <https://github.com/mreineck/pocketfft> |
| MPICH 4.2.3 and the modules it embeds (`-mpi` image)               | MPICH license (BSD-style)               | <https://www.mpich.org/>                |
| GCC runtime libraries                                              | GPL-3.0-or-later WITH GCC-exception-3.1 | <https://gcc.gnu.org/>                  |
| GNU C Library                                                      | LGPL-2.1-or-later                       | <https://www.gnu.org/software/libc/>    |
| NVIDIA CUDA runtime libraries (`-cuda` image, from its base image) | NVIDIA CUDA Toolkit EULA (proprietary)  | <https://docs.nvidia.com/cuda/eula/>    |
| Ubuntu 24.04 base system                                           | Per package                             | <https://ubuntu.com/>                   |

`CMakeLists.txt` pins the versions of the dependencies compiled into the binaries; the
Ubuntu packages are Ubuntu 24.04's.
