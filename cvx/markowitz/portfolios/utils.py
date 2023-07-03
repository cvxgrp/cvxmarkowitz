# -*- coding: utf-8 -*-
def approx(name, x, target, pm):
    yield f"{name}_approx_upper", x - target <= pm
    yield f"{name}_approx_lower", x - target >= -pm
