# *********************************************************************************************************** #
#                                                                                                             #
#                                                               :::::::::: ::::::::   :::::::: :::::::::::    #
#    main.py                                                   :+:       :+:    :+: :+:    :+:    :+:         #
#                                                             +:+       +:+        +:+           +:+          #
#    By: YA. Adam <adam.y.abdc@gmail.com>                    +#++:++#  +#++:++#++ :#:           +#+           #
#                                                           +#+              +#+ +#+   +#+#    +#+            #
#    Created: 2024/03/18 21:40:46 by YA. Adam              #+#       #+#    #+# #+#    #+#    #+#             #
#    Updated: 2024/03/18 21:40:46 by YA. Adam             ########## ########   ######## ###########          #
#                                                                                                             #
# *********************************************************************************************************** #

from ctypes import cdll

if __name__ == "__main__":
    lib_path = "./../mylib/target/debug/libmylib.so"
    lib = cdll.LoadLibrary(lib_path)
    lib.hello()
    sp = lib.multilayer_perceptron.singlelayer_perceptron.singlelayer_perceptron.new()
    sp.to_string()
