
class GlobalResult:
    redhit_result = []
    garak_result = []

    @classmethod
    def add_result(cls, redhit_result, garak_result):

        if redhit_result is not None:
            cls.redhit_result.append(redhit_result)

        if garak_result is not None:
            cls.garak_result.append(garak_result)
