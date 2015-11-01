# ROOTDIR should point to the projects directory where the NVIDIA CUDA SDK was installed.
# Since we don't depend on the CUDA SDK, this shouldn't matter.
ROOTDIR=.

# ROOTBINDIR set to override the binary output directory so they get placed here
# rather than into the CUDA SDK projects directory.
ROOTBINDIR=./bin

<<<<<<< HEAD
CUDACCFLAGS=-Xcompiler -O0
=======
#CUDACCFLAGS=-Xopencc -OPT:Olimit=0
#CUDACCFLAGS=-OPT:Olimit=0
>>>>>>> 926cc8259b15e877d01171cb52de2af73a6a5fea
