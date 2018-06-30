cuda_add_executable(vulcan-bin vulcan.cu)
target_link_libraries(vulcan-bin vulcan)
set_target_properties(vulcan-bin PROPERTIES OUTPUT_NAME vulcan)