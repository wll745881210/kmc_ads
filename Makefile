############################################################
# Compiler and Linker
CC          := nvcc

# The Target Binary Program
TARGET      := ads_emu

SRCDIR      := src
USRDIR      := usr
INCDIR      :=
BUILDDIR    := obj
TARGETDIR   := bin
SRCEXT      := cpp
DEPEXT      := d
OBJEXT      := o

CFLAGS_CMD  :=

# Flags, Libraries and Includes
ifeq ($(DEBUG),1) 
CFLAGS      := -std=c++17 -rdc=true -arch=sm_80 -O0 -g -G \
               -include ./src/utilities/debug_macros.h \
	           -D __GPU_DEBUG__
else
CFLAGS      := -std=c++17 -rdc=true -arch=sm_80 -O3 \
               --use_fast_math $(CFLAGS_CMD)
endif
LIB         := $(CFLAGS)
INC         := -x cu
INCDEP      :=

ifeq ($(MPI),1) 
CFLAGS      += -D__MPI__
LIB         += -lmpi
endif

############################################################
# Automatic generating objs and deps

SOURCES := $(shell find -L $(SRCDIR) -type f \
             -name *.$(SRCEXT))
OBJECTS := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,\
            $(SOURCES:.$(SRCEXT)=.$(OBJEXT)))
USRSRC  := $(shell find -L $(USRDIR) -type f \
             -name *.$(SRCEXT))
OBJECTS += $(patsubst $(USRDIR)/%,$(BUILDDIR)/$(USRDIR)/%,\
            $(USRSRC:.$(SRCEXT)=.$(OBJEXT)))

.PHONY : all dirs clean remake print

#Defauilt Make
all: dirs $(TARGET)

# print : ; $(info $$(SOURCES) is [${SOURCES}])
print : ; $(info $$(OBJECTS) is [${OBJECTS}])

# Remake
remake: clean all

# Make the Directories
dirs:
	@mkdir -p $(TARGETDIR)
	@mkdir -p $(BUILDDIR)
	@mkdir -p $(BUILDDIR)/$(USRDIR)

#C lean only Objecst
clean:
	@$(RM) -rf $(BUILDDIR)/* $(TARGETDIR)/*

# Pull in dependency info for *existing* .o files
-include $(OBJECTS:.$(OBJEXT)=.$(DEPEXT))

# Link
$(TARGET): $(OBJECTS)
	$(CC) -o $(TARGETDIR)/$(TARGET) $^ $(LIB)

# Compile
$(BUILDDIR)/%.$(OBJEXT): $(SRCDIR)/%.$(SRCEXT)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
	@$(CC) $(CFLAGS) $(INCDEP) -M $(SRCDIR)/$*.$(SRCEXT) >\
        $(BUILDDIR)/$*.$(DEPEXT)
	@cp -f $(BUILDDIR)/$*.$(DEPEXT) \
           $(BUILDDIR)/$*.$(DEPEXT).tmp
	@sed -e 's|.*:|$(BUILDDIR)/$*.$(OBJEXT):|' \
         < $(BUILDDIR)/$*.$(DEPEXT).tmp \
         > $(BUILDDIR)/$*.$(DEPEXT)
	@sed -e 's/.*://' -e 's/\\$$//' \
          < $(BUILDDIR)/$*.$(DEPEXT).tmp | fmt -1 | \
          sed -e 's/^ *//' -e 's/$$/:/' >> \
          $(BUILDDIR)/$*.$(DEPEXT)
	@rm -f $(BUILDDIR)/$*.$(DEPEXT).tmp

$(BUILDDIR)/$(USRDIR)/%.$(OBJEXT): $(USRDIR)/%.$(SRCEXT)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@
	@$(CC) $(CFLAGS) $(INCDEP) -M $(USRDIR)/$*.$(SRCEXT) >\
        $(BUILDDIR)/$(USRDIR)/$*.$(DEPEXT)
	@cp -f $(BUILDDIR)/$(USRDIR)/$*.$(DEPEXT) \
           $(BUILDDIR)/$(USRDIR)/$*.$(DEPEXT).tmp
	@sed -e 's|.*:|$(BUILDDIR)/$(USRDIR)/$*.$(OBJEXT):|' \
         < $(BUILDDIR)/$(USRDIR)/$*.$(DEPEXT).tmp \
         > $(BUILDDIR)/$(USRDIR)/$*.$(DEPEXT)
	@sed -e 's/.*://' -e 's/\\$$//' \
        < $(BUILDDIR)/$(USRDIR)/$*.$(DEPEXT).tmp | fmt -1 |\
          sed -e 's/^ *//' -e 's/$$/:/' >> \
          $(BUILDDIR)/$(USRDIR)/$*.$(DEPEXT)
	@rm -f $(BUILDDIR)/$(USRDIR)/$*.$(DEPEXT).tmp
