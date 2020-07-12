# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/kunal/g4work/XFCT_Main/Components/Debugging/convolve

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/kunal/g4work/XFCT_Main/Components/Debugging/convolve/build

# Include any dependencies generated for this target.
include CMakeFiles/fftConvolve.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/fftConvolve.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/fftConvolve.dir/flags.make

CMakeFiles/fftConvolve.dir/fftConvolve.cc.o: CMakeFiles/fftConvolve.dir/flags.make
CMakeFiles/fftConvolve.dir/fftConvolve.cc.o: ../fftConvolve.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/kunal/g4work/XFCT_Main/Components/Debugging/convolve/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/fftConvolve.dir/fftConvolve.cc.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/fftConvolve.dir/fftConvolve.cc.o -c /home/kunal/g4work/XFCT_Main/Components/Debugging/convolve/fftConvolve.cc

CMakeFiles/fftConvolve.dir/fftConvolve.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fftConvolve.dir/fftConvolve.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/kunal/g4work/XFCT_Main/Components/Debugging/convolve/fftConvolve.cc > CMakeFiles/fftConvolve.dir/fftConvolve.cc.i

CMakeFiles/fftConvolve.dir/fftConvolve.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fftConvolve.dir/fftConvolve.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/kunal/g4work/XFCT_Main/Components/Debugging/convolve/fftConvolve.cc -o CMakeFiles/fftConvolve.dir/fftConvolve.cc.s

CMakeFiles/fftConvolve.dir/fftConvolve.cc.o.requires:

.PHONY : CMakeFiles/fftConvolve.dir/fftConvolve.cc.o.requires

CMakeFiles/fftConvolve.dir/fftConvolve.cc.o.provides: CMakeFiles/fftConvolve.dir/fftConvolve.cc.o.requires
	$(MAKE) -f CMakeFiles/fftConvolve.dir/build.make CMakeFiles/fftConvolve.dir/fftConvolve.cc.o.provides.build
.PHONY : CMakeFiles/fftConvolve.dir/fftConvolve.cc.o.provides

CMakeFiles/fftConvolve.dir/fftConvolve.cc.o.provides.build: CMakeFiles/fftConvolve.dir/fftConvolve.cc.o


# Object files for target fftConvolve
fftConvolve_OBJECTS = \
"CMakeFiles/fftConvolve.dir/fftConvolve.cc.o"

# External object files for target fftConvolve
fftConvolve_EXTERNAL_OBJECTS =

fftConvolve: CMakeFiles/fftConvolve.dir/fftConvolve.cc.o
fftConvolve: CMakeFiles/fftConvolve.dir/build.make
fftConvolve: /usr/lib/x86_64-linux-gnu/libfftw3f.so.3
fftConvolve: /usr/lib/x86_64-linux-gnu/libfftw3.so.3
fftConvolve: /usr/lib/x86_64-linux-gnu/libfftw3_threads.so.3
fftConvolve: CMakeFiles/fftConvolve.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/kunal/g4work/XFCT_Main/Components/Debugging/convolve/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable fftConvolve"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/fftConvolve.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/fftConvolve.dir/build: fftConvolve

.PHONY : CMakeFiles/fftConvolve.dir/build

CMakeFiles/fftConvolve.dir/requires: CMakeFiles/fftConvolve.dir/fftConvolve.cc.o.requires

.PHONY : CMakeFiles/fftConvolve.dir/requires

CMakeFiles/fftConvolve.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/fftConvolve.dir/cmake_clean.cmake
.PHONY : CMakeFiles/fftConvolve.dir/clean

CMakeFiles/fftConvolve.dir/depend:
	cd /home/kunal/g4work/XFCT_Main/Components/Debugging/convolve/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/kunal/g4work/XFCT_Main/Components/Debugging/convolve /home/kunal/g4work/XFCT_Main/Components/Debugging/convolve /home/kunal/g4work/XFCT_Main/Components/Debugging/convolve/build /home/kunal/g4work/XFCT_Main/Components/Debugging/convolve/build /home/kunal/g4work/XFCT_Main/Components/Debugging/convolve/build/CMakeFiles/fftConvolve.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/fftConvolve.dir/depend

