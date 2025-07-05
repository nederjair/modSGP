
class Methods:
    @staticmethod
    def euler(xi_minus_1, ui_minus_1, h, model):
        dx_dt = model(xi_minus_1, ui_minus_1)
        xi = xi_minus_1 + h * dx_dt
        return xi

    @staticmethod
    def heun(xi_minus_1, ui_minus_1, ui, h, model):
        dx_dt = model(xi_minus_1, ui_minus_1)
        xi_minus_1_star = xi_minus_1 + h * dx_dt
        ui_minus_1_star = (ui_minus_1 + ui) / 2
        dx_dt_star = model(xi_minus_1_star, ui_minus_1_star)
        xi = xi_minus_1 + (h / 2) * (dx_dt + dx_dt_star)
        return xi
