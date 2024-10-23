#include <cstring>

#include "aimrt_pkg_c_interface/pkg_macro.h"
#include "dcu_driver_module/dcu_driver_module.h"
#include "joy_stick_module/joy_stick_module.h"
#include "rl_control_module/rl_control_module.h"

static std::tuple<std::string_view, std::function<aimrt::ModuleBase*()>>
    aimrt_module_register_array[]{
        {"DcuDriverModule",
         []() -> aimrt::ModuleBase* {
           return new xyber_x1_infer::dcu_driver_module::DcuDriverModule();
         }},
        {"JoyStickModule",
         []() -> aimrt::ModuleBase* {
           return new xyber_x1_infer::joy_stick_module::JoyStickModule();
         }},
        {"RlControlModule",
         []() -> aimrt::ModuleBase* {
           return new xyber_x1_infer::rl_control_module::RlControlModule();
         }},
    };

AIMRT_PKG_MAIN(aimrt_module_register_array)
