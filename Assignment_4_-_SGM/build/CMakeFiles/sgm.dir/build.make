# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.27

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/homebrew/Cellar/cmake/3.27.5/bin/cmake

# The command to remove a file.
RM = /opt/homebrew/Cellar/cmake/3.27.5/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/alejandrocampayofernandez/Desktop/Saarland/saarland-assignments/3DCV/Assignment_4_-_SGM/Assignment4

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/alejandrocampayofernandez/Desktop/Saarland/saarland-assignments/3DCV/Assignment_4_-_SGM/Assignment4/build

# Include any dependencies generated for this target.
include CMakeFiles/sgm.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/sgm.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/sgm.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/sgm.dir/flags.make

CMakeFiles/sgm.dir/sgm.cpp.o: CMakeFiles/sgm.dir/flags.make
CMakeFiles/sgm.dir/sgm.cpp.o: /Users/alejandrocampayofernandez/Desktop/Saarland/saarland-assignments/3DCV/Assignment_4_-_SGM/Assignment4/sgm.cpp
CMakeFiles/sgm.dir/sgm.cpp.o: CMakeFiles/sgm.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/alejandrocampayofernandez/Desktop/Saarland/saarland-assignments/3DCV/Assignment_4_-_SGM/Assignment4/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/sgm.dir/sgm.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/sgm.dir/sgm.cpp.o -MF CMakeFiles/sgm.dir/sgm.cpp.o.d -o CMakeFiles/sgm.dir/sgm.cpp.o -c /Users/alejandrocampayofernandez/Desktop/Saarland/saarland-assignments/3DCV/Assignment_4_-_SGM/Assignment4/sgm.cpp

CMakeFiles/sgm.dir/sgm.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/sgm.dir/sgm.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/alejandrocampayofernandez/Desktop/Saarland/saarland-assignments/3DCV/Assignment_4_-_SGM/Assignment4/sgm.cpp > CMakeFiles/sgm.dir/sgm.cpp.i

CMakeFiles/sgm.dir/sgm.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/sgm.dir/sgm.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/alejandrocampayofernandez/Desktop/Saarland/saarland-assignments/3DCV/Assignment_4_-_SGM/Assignment4/sgm.cpp -o CMakeFiles/sgm.dir/sgm.cpp.s

# Object files for target sgm
sgm_OBJECTS = \
"CMakeFiles/sgm.dir/sgm.cpp.o"

# External object files for target sgm
sgm_EXTERNAL_OBJECTS =

libsgm.dylib: CMakeFiles/sgm.dir/sgm.cpp.o
libsgm.dylib: CMakeFiles/sgm.dir/build.make
libsgm.dylib: CMakeFiles/sgm.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/alejandrocampayofernandez/Desktop/Saarland/saarland-assignments/3DCV/Assignment_4_-_SGM/Assignment4/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library libsgm.dylib"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/sgm.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/sgm.dir/build: libsgm.dylib
.PHONY : CMakeFiles/sgm.dir/build

CMakeFiles/sgm.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/sgm.dir/cmake_clean.cmake
.PHONY : CMakeFiles/sgm.dir/clean

CMakeFiles/sgm.dir/depend:
	cd /Users/alejandrocampayofernandez/Desktop/Saarland/saarland-assignments/3DCV/Assignment_4_-_SGM/Assignment4/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/alejandrocampayofernandez/Desktop/Saarland/saarland-assignments/3DCV/Assignment_4_-_SGM/Assignment4 /Users/alejandrocampayofernandez/Desktop/Saarland/saarland-assignments/3DCV/Assignment_4_-_SGM/Assignment4 /Users/alejandrocampayofernandez/Desktop/Saarland/saarland-assignments/3DCV/Assignment_4_-_SGM/Assignment4/build /Users/alejandrocampayofernandez/Desktop/Saarland/saarland-assignments/3DCV/Assignment_4_-_SGM/Assignment4/build /Users/alejandrocampayofernandez/Desktop/Saarland/saarland-assignments/3DCV/Assignment_4_-_SGM/Assignment4/build/CMakeFiles/sgm.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/sgm.dir/depend

