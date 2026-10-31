#
# // Copyright (c) Radzivon Bartoshyk 5/2026. All rights reserved.
# //
# // Redistribution and use in source and binary forms, with or without modification,
# // are permitted provided that the following conditions are met:
# //
# // 1.  Redistributions of source code must retain the above copyright notice, this
# // list of conditions and the following disclaimer.
# //
# // 2.  Redistributions in binary form must reproduce the above copyright notice,
# // this list of conditions and the following disclaimer in the documentation
# // and/or other materials provided with the distribution.
# //
# // 3.  Neither the name of the copyright holder nor the names of its
# // contributors may be used to endorse or promote products derived from
# // this software without specific prior written permission.
# //
# // THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# // AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# // IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# // DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# // FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# // DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# // SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# // CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# // OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# // OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

cargo run --package stats --release -- ./assets/train0/00004_TE_1808x1352.png --no-image-avif --distances 0.03,0.15,0.25,0.295,0.3,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6 --efforts 3,6 --out charts --log

cargo run --package stats --release -- ./assets/Kodak/01.png --distances 0.03,0.15,0.25,0.295,0.3,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9 --efforts 3,6 --out charts --butteraugli --log

cargo run --package stats --release -- ./assets/Kodak/05.png --no-image-avif --distances 0.03,0.15,0.25,0.295,0.3,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9 --efforts 3,6 --out charts --butteraugli --cvvdp --arbiter cvvdp --log

cargo run --package stats --release -- ./assets/volcanic.png --no-image-avif --distances 0.03,0.15,0.25,0.295,0.3,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9 --efforts 3,6 --out charts --butteraugli --cvvdp --arbiter cvvdp --log

cargo run --package stats --release -- ./assets/volcanic.png --no-av2 --no-image-avif --hevc --distances 0.15,0.25,0.295,0.3,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9 --efforts 7 --out charts --butteraugli --cvvdp --log

cargo run --package stats --release -- "./assets/illustrations/jam-reference.png" --no-image-avif --no-av2 --distances 0.03,0.15,0.25,0.295,0.3,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9 --efforts 3,6 --out charts --butteraugli --cvvdp --log

cargo run --package stats --release -- "./assets/illustrations/jam-reference.png" --no-image-avif --distances 0.3,0.5,1,1.5,2,2.5 --efforts 6 --out charts --butteraugli --cvvdp --log

cargo run --package stats --release -- ./assets/volcanic.png --no-av2 --distances 0.15,0.25,0.295,0.3,0.5,1,1.25,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9 --efforts 3,6 --out charts --butteraugli --cvvdp --arbiter cvvdp --log

cargo run --package stats --release -- ./assets/inadra.png --no-image-avif --distances 0.03,0.15,0.25,0.295,0.3,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9 --efforts 3,6 --out charts --butteraugli --cvvdp --arbiter cvvdp --log

cargo run --package stats --release -- ./assets/inadra.png --no-av2 --no-image-avif --distances 0.3,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9 --efforts 3,6 --out charts --butteraugli --cvvdp --arbiter cvvdp --log

cargo run --package stats --release -- ./assets/jyrki31/geranium.png --distances 0.03,0.15,0.25,0.295,0.3,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9 --efforts 3,6 --out charts --butteraugli --log

cargo run --package stats --release -- ./assets/137237150_p0.png --no-av2 --no-image-avif --distances 0.1,0.15,0.25,0.295,0.3,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5 --efforts 3,6 --out charts --butteraugli --cvvdp --arbiter cvvdp --log

cargo run --package stats --release -- ./assets/jyrki31/geranium.png --no-av2 --no-image-avif --distances 0.03,0.15,0.25,0.295,0.3,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9 --efforts 3,6 --out charts --butteraugli --cvvdp --arbiter cvvdp --log

cargo run --package stats --release -- ./assets/Kodak/10.png --no-av2 --no-image-avif --distances 0.03,0.15,0.25,0.295,0.3,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9 --efforts 3,6 --out charts --butteraugli --cvvdp --arbiter cvvdp --log

cargo run --package stats --release -- ./assets/sun/pexels-anatoleos-35404883.jpg --distances 0.03,0.15,0.25,0.295,0.3,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9 --efforts 3,6 --out charts --butteraugli --log

cargo run --package stats --release -- ./assets/Burning_Ship_Fractal.png --no-av2 --no-image-avif --distances 0.03,0.15,0.25,0.295,0.3,0.5,1,1.25,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6 --efforts 3,6 --out charts --butteraugli --cvvdp --log

cargo run --package stats --release -- ./assets/nasa/web-first-images-release.png --no-av2 --no-image-avif --distances 0.03,0.15,0.25,0.295,0.3,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6 --efforts 3,7 --out charts --butteraugli --log --cvvdp --arbiter cvvdp

cargo run --package stats --release -- ./assets/yellow/pexels-david-underland-6726219.jpg --no-av2 --no-image-avif --distances 0.03,0.15,0.25,0.295,0.3,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6 --efforts 3,6 --out charts --butteraugli --log

cargo run --package stats --release -- ./assets/volcanic.png --no-aom --no-image-avif --distances 0.03,0.15,0.25,0.295,0.3,0.5,1,1.5,2,2.5,3 --efforts 3,7 --out charts --butteraugli --log --cvddp

cargo run -p meanstats --release -- ./assets/train0 --distances 0.15,0.25,0.295,0.3,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6 --threads 12 --log

cargo run -p meanstats --release -- ./assets/train0 --distances 0.15,0.25,0.295,0.3,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6 --efforts 3,7 --threads 12 --log

cargo run -p meanstats --release -- ./assets/Kodak --distances 0.03,0.15,0.25,0.295,0.3,0.5,1,1.25,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6 --efforts 3,7 --threads 12 --cvvdp --no-butteraugli --log

cargo run -p meanstats --release -- ./assets/jyrki31 --distances 0.03,0.15,0.25,0.295,0.3,0.5,1,1.25,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6 --efforts 3,7 --threads 12 --cvvdp --no-butteraugli --log

cargo run -p meanstats --release -- ./assets/Kodak --distances 0.75,0.85,1,1.25,1.5,1.75,2,2.25,2.5,3,3.5,3.75,4,4.25 --efforts 3,7 --threads 12 --cvvdp --no-butteraugli --log

cargo run -p meanstats --release -- ./assets/jpeg_xl_png --distances 0.03,0.15,0.25,0.295,0.3,0.5,0.75,1,1.25,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6 --efforts 3,7 --threads 12 --cvvdp --log
