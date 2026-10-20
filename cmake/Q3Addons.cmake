# Build-time add-ons.
#
# An add-on is a directory src/addons/<name>/ holding a CMakeLists.txt. It
# extends the core only through the core's registries (input formats,
# pseudopotential formats, PAW one-centre conventions, volumetric formats), so
# the core names no add-on anywhere: a tree without src/addons/ builds and runs
# unchanged, and dropping an add-on directory into the tree compiles it in.
#
# Each add-on's CMakeLists.txt calls q3_register_addon() once. q3_build_addons()
# then writes two headers into the generated include directory:
#
#   q3_addon_identity.hpp      the active add-ons' names and versions, as data;
#                              read by layer_0_core/addon_identity.hpp for
#                              `--version` and the results provenance block.
#   q3_addon_registration.hpp  includes every add-on's registration header and
#                              defines q3::addons::register_all(), which each
#                              entry point calls before it reads input; read
#                              through io/addon_registration.hpp.
#
# Registration is an explicit call rather than static initialisation, so it
# cannot depend on the order in which translation units initialise.

define_property(GLOBAL PROPERTY Q3_ADDON_NAMES
    BRIEF_DOCS "Names of the add-ons compiled into this build, in discovery order"
    FULL_DOCS "Appended to by q3_register_addon().")

# Register one add-on. Called from src/addons/<name>/CMakeLists.txt.
#
#   NAME                   the add-on's name, as --version and results.json report it
#   VERSION                the add-on's version
#   REGISTRATION_HEADER    the header declaring the registration function, relative to src/
#   REGISTRATION_FUNCTION  the fully qualified function that registers everything the add-on adds
#   UNIT_TESTS             the add-on's GTest sources, relative to the repository root;
#                          built into q3_tests when tests are enabled
function(q3_register_addon)
    cmake_parse_arguments(ADDON ""
        "NAME;VERSION;REGISTRATION_HEADER;REGISTRATION_FUNCTION" "UNIT_TESTS" ${ARGN})
    foreach(required NAME VERSION REGISTRATION_HEADER REGISTRATION_FUNCTION)
        if(NOT ADDON_${required})
            message(FATAL_ERROR "q3_register_addon(): ${required} is required")
        endif()
    endforeach()
    get_property(registered_names GLOBAL PROPERTY Q3_ADDON_NAMES)
    if(ADDON_NAME IN_LIST registered_names)
        message(FATAL_ERROR "q3_register_addon(): the add-on '${ADDON_NAME}' is registered twice")
    endif()

    set_property(GLOBAL APPEND PROPERTY Q3_ADDON_NAMES "${ADDON_NAME}")
    set_property(GLOBAL PROPERTY Q3_ADDON_VERSION_${ADDON_NAME} "${ADDON_VERSION}")
    set_property(GLOBAL PROPERTY Q3_ADDON_HEADER_${ADDON_NAME} "${ADDON_REGISTRATION_HEADER}")
    set_property(GLOBAL PROPERTY Q3_ADDON_FUNCTION_${ADDON_NAME} "${ADDON_REGISTRATION_FUNCTION}")

    if(TARGET q3_tests AND ADDON_UNIT_TESTS)
        list(TRANSFORM ADDON_UNIT_TESTS PREPEND "${PROJECT_SOURCE_DIR}/")
        target_sources(q3_tests PRIVATE ${ADDON_UNIT_TESTS})
    endif()
endfunction()

# Discover and build every add-on present in the tree, then write the two
# generated headers. Called once from the root CMakeLists.txt, after the test
# targets exist so an add-on can add its tests to them.
function(q3_build_addons)
    file(GLOB addon_manifests LIST_DIRECTORIES false
        "${PROJECT_SOURCE_DIR}/src/addons/*/CMakeLists.txt")
    list(SORT addon_manifests)
    foreach(manifest IN LISTS addon_manifests)
        get_filename_component(addon_directory "${manifest}" DIRECTORY)
        add_subdirectory("${addon_directory}")
    endforeach()

    get_property(addon_names GLOBAL PROPERTY Q3_ADDON_NAMES)
    set(Q3_ADDON_IDENTITY_ENTRIES "")
    set(Q3_ADDON_REGISTRATION_INCLUDES "")
    set(Q3_ADDON_REGISTRATION_CALLS "")
    set(summary "")
    foreach(addon_name IN LISTS addon_names)
        get_property(addon_version GLOBAL PROPERTY Q3_ADDON_VERSION_${addon_name})
        get_property(addon_header GLOBAL PROPERTY Q3_ADDON_HEADER_${addon_name})
        get_property(addon_function GLOBAL PROPERTY Q3_ADDON_FUNCTION_${addon_name})
        string(APPEND Q3_ADDON_IDENTITY_ENTRIES "{\"${addon_name}\", \"${addon_version}\"},")
        string(APPEND Q3_ADDON_REGISTRATION_INCLUDES "#include \"${addon_header}\"\n")
        string(APPEND Q3_ADDON_REGISTRATION_CALLS "        ${addon_function}();\n")
        string(APPEND summary " ${addon_name} ${addon_version}")
    endforeach()

    configure_file("${PROJECT_SOURCE_DIR}/cmake/q3_addon_identity.hpp.in"
        "${Q3_GENERATED_INCLUDE_DIR}/q3_addon_identity.hpp" @ONLY)
    configure_file("${PROJECT_SOURCE_DIR}/cmake/q3_addon_registration.hpp.in"
        "${Q3_GENERATED_INCLUDE_DIR}/q3_addon_registration.hpp" @ONLY)
    if(addon_names)
        message(STATUS "Add-ons compiled in:${summary}")
    else()
        message(STATUS "Add-ons compiled in: none")
    endif()
endfunction()
