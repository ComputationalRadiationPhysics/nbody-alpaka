# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.3

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
CMAKE_SOURCE_DIR = /home/vincent/Dokumente/alpaka/nbody-alpaka/tests/updatePositionsKernel

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/vincent/Dokumente/alpaka/nbody-alpaka/tests/updatePositionsKernel

# Include any dependencies generated for this target.
include CMakeFiles/updatePositions_test.out.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/updatePositions_test.out.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/updatePositions_test.out.dir/flags.make

CMakeFiles/updatePositions_test.out.dir/updatePositions_test.cpp.o: CMakeFiles/updatePositions_test.out.dir/flags.make
CMakeFiles/updatePositions_test.out.dir/updatePositions_test.cpp.o: updatePositions_test.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/vincent/Dokumente/alpaka/nbody-alpaka/tests/updatePositionsKernel/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/updatePositions_test.out.dir/updatePositions_test.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/updatePositions_test.out.dir/updatePositions_test.cpp.o -c /home/vincent/Dokumente/alpaka/nbody-alpaka/tests/updatePositionsKernel/updatePositions_test.cpp

CMakeFiles/updatePositions_test.out.dir/updatePositions_test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/updatePositions_test.out.dir/updatePositions_test.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/vincent/Dokumente/alpaka/nbody-alpaka/tests/updatePositionsKernel/updatePositions_test.cpp > CMakeFiles/updatePositions_test.out.dir/updatePositions_test.cpp.i

CMakeFiles/updatePositions_test.out.dir/updatePositions_test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/updatePositions_test.out.dir/updatePositions_test.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/vincent/Dokumente/alpaka/nbody-alpaka/tests/updatePositionsKernel/updatePositions_test.cpp -o CMakeFiles/updatePositions_test.out.dir/updatePositions_test.cpp.s

CMakeFiles/updatePositions_test.out.dir/updatePositions_test.cpp.o.requires:

.PHONY : CMakeFiles/updatePositions_test.out.dir/updatePositions_test.cpp.o.requires

CMakeFiles/updatePositions_test.out.dir/updatePositions_test.cpp.o.provides: CMakeFiles/updatePositions_test.out.dir/updatePositions_test.cpp.o.requires
	$(MAKE) -f CMakeFiles/updatePositions_test.out.dir/build.make CMakeFiles/updatePositions_test.out.dir/updatePositions_test.cpp.o.provides.build
.PHONY : CMakeFiles/updatePositions_test.out.dir/updatePositions_test.cpp.o.provides

CMakeFiles/updatePositions_test.out.dir/updatePositions_test.cpp.o.provides.build: CMakeFiles/updatePositions_test.out.dir/updatePositions_test.cpp.o


# Object files for target updatePositions_test.out
updatePositions_test_out_OBJECTS = \
"CMakeFiles/updatePositions_test.out.dir/updatePositions_test.cpp.o"

# External object files for target updatePositions_test.out
updatePositions_test_out_EXTERNAL_OBJECTS =

updatePositions_test.out: CMakeFiles/updatePositions_test.out.dir/updatePositions_test.cpp.o
updatePositions_test.out: CMakeFiles/updatePositions_test.out.dir/build.make
updatePositions_test.out: /usr/lib/x86_64-linux-gnu/libboost_unit_test_framework.so
updatePositions_test.out: CMakeFiles/updatePositions_test.out.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/vincent/Dokumente/alpaka/nbody-alpaka/tests/updatePositionsKernel/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable updatePositions_test.out"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/updatePositions_test.out.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/updatePositions_test.out.dir/build: updatePositions_test.out

.PHONY : CMakeFiles/updatePositions_test.out.dir/build

CMakeFiles/updatePositions_test.out.dir/requires: CMakeFiles/updatePositions_test.out.dir/updatePositions_test.cpp.o.requires

.PHONY : CMakeFiles/updatePositions_test.out.dir/requires

CMakeFiles/updatePositions_test.out.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/updatePositions_test.out.dir/cmake_clean.cmake
.PHONY : CMakeFiles/updatePositions_test.out.dir/clean

CMakeFiles/updatePositions_test.out.dir/depend:
	cd /home/vincent/Dokumente/alpaka/nbody-alpaka/tests/updatePositionsKernel && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/vincent/Dokumente/alpaka/nbody-alpaka/tests/updatePositionsKernel /home/vincent/Dokumente/alpaka/nbody-alpaka/tests/updatePositionsKernel /home/vincent/Dokumente/alpaka/nbody-alpaka/tests/updatePositionsKernel /home/vincent/Dokumente/alpaka/nbody-alpaka/tests/updatePositionsKernel /home/vincent/Dokumente/alpaka/nbody-alpaka/tests/updatePositionsKernel/CMakeFiles/updatePositions_test.out.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/updatePositions_test.out.dir/depend

