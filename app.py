import streamlit as st
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Any
import scipy
import sympy
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Ряд Фур'є для ПФЗП")

def abs_sin_exp(x: np.ndarray, alpha: float) -> np.ndarray:
    return np.abs(np.sin(x ** alpha))

def sign_sin_exp(x: np.ndarray, alpha: float) -> np.ndarray:
    return np.sign(np.sin(x ** alpha))

def power(x: np.ndarray, alpha: float) -> np.ndarray:
    return x ** alpha

def power_derivative(x: np.ndarray, alpha) -> np.ndarray:
    return alpha * x ** (alpha - 1)

def power_inverse(x: np.ndarray, alpha: float) -> np.ndarray:
    return x ** (1 / alpha)

def T(x: float, alpha: float) -> float:
    return -x + func.g_inverse(func.g(x, alpha) + 2 * np.pi, alpha)


def modf_exp(x: np.ndarray, alpha: float) -> np.ndarray:
    return np.modf(x ** alpha)[0]

FunctionSpec = Tuple[Callable, Callable, Callable] # (g(x), g^(-1)(x), g'(x))

power_spec = (power, power_inverse, power_derivative)

@dataclass(frozen=True)
class Function:
    full_function: Callable
    name: str
    latex: str
    g: Callable
    g_inverse: Callable
    g_derivative: Callable

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.full_function(*args, **kwds)
    
    @staticmethod
    def manual_entry() -> 'Function':
        return Function(
            full_function=lambda:None,
            name="Ввести вручну в форматі SymPy",
            latex="",
            g=lambda:None,
            g_derivative=lambda:None,
            g_inverse=lambda:None)


functions: List[Function] = [
    Function(
        full_function=abs_sin_exp,
        name=r"|sin(xᵅ)|",
        latex=r"|\sin(x^{\alpha})|",
        g=power,
        g_inverse=power_inverse,
        g_derivative=power_derivative,
    ),
    Function(
        full_function=sign_sin_exp,
        name=r"sign(sin(xᵅ))",
        latex=r"\mathrm{sign}(\sin(x^{\alpha}))",
        g=power,
        g_inverse=power_inverse,
        g_derivative=power_derivative,
    ),
    Function(
        full_function=modf_exp,
        name=r"{xᵅ}",
        latex=r"\{x^\alpha\}",
        g=power,
        g_inverse=power_inverse,
        g_derivative=power_derivative,
    ),
    Function.manual_entry(),
]


col1, col2 = st.columns(2)


def get_function() -> Function:
    func_index = st.radio(
        "Періодична функція $f(x)$ зі змінним періодом",
        options=list(range(len(functions))),
        format_func=lambda idx: functions[idx].name,
    )

    enter_function_manually = func_index == len(functions) - 1

    if enter_function_manually:
        try:
            func = make_manual_function_input()
        except:
            func = None
    else:
        func = functions[func_index] if func_index else functions[0]
    return func

def sanitize_expr_str(expr: str) -> str:
    # TODO: add some more useful functions
    return expr.replace("^", "**")


def make_manual_function_input() -> Function:
    available_syntax_text = r"Підтримуються змінні $x$ (вводити `x`) та $\alpha$ (вводити `alpha`), арифметичні операції, `sign`, `sqrt`, `^`, `abs`, тригонометричні функції"
    x, alpha = sympy.symbols("x alpha")
    full_func_expression = make_function_expression_input(
        label=("**Вираз для функції $f(x)$ в форматі SymPy**\n\n" + available_syntax_text),
        value="sign(sin(x^alpha))",
    )
    g_expression = make_function_expression_input(
        func_def_name_latex="g(x)",
        label=("**Вираз для функції $g(x)$ в форматі SymPy**\n\n" + available_syntax_text),
        value="x^alpha",
    )
    g_derivative_expression = make_function_expression_input(
        func_def_name_latex="g'(x)",
        label=("**Вираз для функції $g'(x)$ в форматі SymPy**\n\n" + available_syntax_text),
        value="alpha * x^(alpha - 1)",
    )
    g_inverse_expression = make_function_expression_input(
        func_def_name_latex="g^{-1}(x)",
        label=("**Вираз для функції $g^{-1}(x)$ в форматі SymPy**\n\n" + available_syntax_text),
        value="x^(1/alpha)",
    )
    full_func = sympy.lambdify([x, alpha], full_func_expression)
    g = sympy.lambdify([x, alpha], g_expression)
    g_inverse = sympy.lambdify([x, alpha], g_inverse_expression)
    g_derivative = sympy.lambdify([x, alpha], g_derivative_expression)
    return Function(full_function=full_func, name=str(full_func_expression), latex=sympy.latex(full_func_expression), g=g, g_derivative=g_derivative, g_inverse=g_inverse)
    
def make_function_expression_input(func_def_name_latex = "f(x)", **kwargs) -> Optional[sympy.Expr]:
    """Make a text input which will parse sympy function of x, alpha and will display error when parsing fails"""
    func_text = st.text_input(**kwargs)
    try:
        expression = sympy.parse_expr(sanitize_expr_str(func_text))
        expression_latex = sympy.latex(expression)
        st.warning(f"Цей вираз інтерпретовано як ${func_def_name_latex}={expression_latex}$")
        return expression
    except:
        st.error("Hе вдалося інтерпретувати вираз")
    

with col1:
    with st.expander("Параметри $f(x)$", expanded=True):
        func = get_function()
        
        alpha_raw = sympy.parse_expr(st.text_input(
            r"$\alpha,\,\alpha > 0$",
            help="Значення у вигляді десяткового числа або дробу",
            value="1/2",  
        ))
        alpha = float(alpha_raw)
        alpha_raw = str(alpha_raw)

    with st.expander("Параметри ряду Фур'є", expanded=True):
        integration_start = float(sympy.parse_expr(st.text_input(
            r"Нижня межа інтегрування $\tau$",
            help="Значення у вигляді десяткового числа або дробу",
            value="0",  
        )).evalf())

        integration_end = T(integration_start, alpha) + integration_start

        st.number_input(
            r"Верхня межа інтегрування $g^{-1}[g(\tau) + 2\pi]$",
            value=integration_end,
            disabled=True,
        )

        n = int(st.number_input(
            r"Кількість доданків ряду Фур'є $n$",
            value=100,
            min_value=1,
        ))

        n_integration = st.number_input(
            "Кількість розбиттів для чисельного інтегрування методом [scipy.integration.quad](https://docs.scipy.org/doc/scipy/tutorial/integrate.html#general-integration-quad)",
            value=100,
            min_value=1,
        )

with col2:        
    with st.expander("Параметри графіка", expanded=True):    
        start_plot = st.number_input(
            "Ліва межа для побудови графіка",
            value=0.0,
            min_value=0.0,
        )

        end_plot = st.number_input(
            "Права межа для побудови графіка",
            value=integration_end,
            min_value=start_plot,
        )

        n_plot = int(st.number_input(
            "Кількість точок для побудови графіка",
            value=500,
            min_value=1,
        ))
      
        plot_titles = st.checkbox("Експортувати назви графіків", value=False)

        manual_axes_ratio = not st.checkbox("Підібрати автоматично співвідношення сторін для графіків", value=False)
        if manual_axes_ratio:
            axes_ratio = 1/float(st.number_input("Співвідношення ширина/висота для графіків",
                                                 value=3.0, min_value=0.01, max_value=8.0))
            st.info(f"Графік буде в **{1/axes_ratio:.1f}** разів ширшим ніж вищим")

        manual_ylim = not st.checkbox("Підібрати автоматично межі $y$ для графіків", value=True)
        if manual_ylim:
            y_min = float(st.number_input("Нижня межа $y_{min}$ для графіків", value=-1.0))
            y_max = float(st.number_input("Верхня межа $y_{max}$ для графіків", value=1.0))

        func_color = st.color_picker("Колір лінії функції $f(x)$", value="#0000FF")
        func_linewidth = st.slider("Товщина лінії функції $f(x)$", value=1., min_value=0.1, max_value=10.)
        func_alpha = 1 - st.slider("Прозорість лінії функції $f(x)$", value=0.05, min_value=0., max_value=1.)
        
        approx_color = st.color_picker("Колір лінії апроксимації", value="#FFA500")
        approx_linewidth = st.slider("Товщина лінії апроксимації", value=1., min_value=0.1, max_value=10.)
        approx_alpha = 1 - st.slider("Прозорість лінії апроксимації", value=0.05, min_value=0., max_value=1.)
            
def integrate(f, start, end) -> float:
    return scipy.integrate.quad(f, start, end, limit=n_integration)[0]

@dataclass(frozen=True)
class FourierSeries:
    a0: float
    an: np.ndarray
    bn: np.ndarray
    alpha: float
    n: int
    limits: Tuple[float, float]
    f: Function

    def __call__(self, x, alpha) -> float:
        result = self.a0 / 2
        for k in range(1, self.n + 1):
            result += self.an[k - 1] * np.cos(k * self.f.g(x, self.alpha))
            result += self.bn[k - 1] * np.sin(k * self.f.g(x, self.alpha))
        return result

    @staticmethod
    def construct(f: Function, alpha: float, n: int, limits: Tuple[float, float]) -> 'FourierSeries':
        start, end = limits
        a0 = integrate(
            lambda x: f.g_derivative(x, alpha) * f(x, alpha),
            start, end
        ) / np.pi
        an = np.array([
            integrate(
                lambda x: f.g_derivative(x, alpha) * f(x, alpha) * np.cos(k * f.g(x, alpha)),
                start, end
            ) / np.pi 
            for k in range(1, n + 1)
        ])
        bn = np.array([
            integrate(
                lambda x: f.g_derivative(x, alpha) * f(x, alpha) * np.sin(k * f.g(x, alpha)),
                start, end
            ) / np.pi 
            for k in range(1, n + 1)
        ])
        return FourierSeries(
            a0=a0, an=an, bn=bn,
            alpha=alpha, n=n, limits=limits,
            f=f,
        )


def plot_function(f: Callable, start_plot: float, end_plot: float, n_points: int, ax: plt.Axes, **kwargs):
    xs = np.linspace(start_plot, end_plot, num=n_points)
    ys = f(xs, alpha)
    ax.set_ylabel(r"$y$")
    ax.set_xlabel(r"$x$")
    ax.grid(True, which='both', color="#eee", linewidth=0.75)
    ax.axhline(y=0, color='#333', linewidth=1)
    ax.axvline(x=0, color='#333', linewidth=1)
    ax.plot(xs, ys, **kwargs)

    if manual_ylim:
        ax.set_ylim([y_min, y_max])

    if manual_axes_ratio:
        fig = plt.gcf()
        w, h = plt.figaspect(axes_ratio)
        fig.set_size_inches(w, h)
        

    
def plot_spectrum(A, ax: plt.Axes, **kwargs):
    ax.set_ylabel(r"$A_k$")
    ax.set_xlabel(r"$k$")
    ax.grid(True, which='both', color="#eee", linewidth=0.75)
    ax.stem(range(1, len(A) + 1), A, **kwargs)
    
def make_coeff_df(series: FourierSeries) -> pd.DataFrame:
    data = np.array([[series.a0] + list(series.an), [0] + list(series.bn)]).T
    index = pd.Index(range(len(data)), name="k")
    df = pd.DataFrame(data=data, columns=["ak", "bk"], index=index)
    return df

def make_spectrum_srs(A: np.ndarray) -> pd.Series:
    index = pd.Index(range(1, len(A) + 1), name="k")
    return pd.Series(A, name="Ak", index=index)

def render_series_info():
    with st.spinner("Зачекайте..."):
        fig = plt.figure()
        ax = plt.subplot()
        plot_function(func,
                      start_plot=start_plot,
                      end_plot=end_plot,
                      n_points=n_plot,
                      label=f"${func.latex}$",
                      color=func_color,
                      linewidth=func_linewidth,
                      alpha=func_alpha,
                      ax=ax)
        series = FourierSeries.construct(f=func,
                                         alpha=alpha,
                                         n=n,
                                         limits=(integration_start, integration_end))
        plot_function(series,
                      start_plot=start_plot,
                      end_plot=end_plot,
                      n_points=n_plot,
                      label=f"Апроксимація ${n=}$",
                      color=approx_color,
                      linewidth=approx_linewidth,
                      alpha=approx_alpha,
                      ax=ax)
        if plot_titles:
            ax.set_title(f"Графік апроксимації ${func.latex}$ "r"($\alpha="f"{alpha_raw}"f"$) рядом Фур'є ${n=}$")
        fig.legend(loc="center right")
        st.subheader(f"Графік апроксимації ${n=}$")
        st.pyplot(fig, clear_figure=False)
        df = make_coeff_df(series=series)
        csv_data = df.to_csv().encode("utf-8")
        st.subheader(f"Коефіцієнти ряду Фур'є для ${n=}$")
        st.download_button(
            "Завантажити таблицю коефіцієнтів $a_k$, $b_k$ в форматі .csv",
            csv_data,
            file_name="ak_bk_coefficients.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.dataframe(df, use_container_width=True)

        st.subheader("Тригонометричні коефіцієнти $A_k = \sqrt{a_k^2 + b_k^2}$")
        An = np.sqrt(series.an ** 2 + series.bn ** 2)
        spectrum = make_spectrum_srs(An)
        csv_spectrum = spectrum.to_csv().encode("utf-8")
        fig, ax = plt.subplots()
        plot_spectrum(An, ax=ax)
        if plot_titles:
            ax.set_title(f"Тригонометричні коефіцієнти $A_k = \sqrt{{a_k^2 + b_k^2}}$ для ряду Фур'є ${n=}$")
        st.pyplot(fig, clear_figure=False)
        st.download_button(
            "Завантажити таблицю коефіцієнтів $A_k$ в форматі .csv",
            csv_spectrum,
            file_name="Ak_coefficients.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.dataframe(spectrum, use_container_width=True)        


with col2:
    with st.form("compute"):
        disabled = func is None
        do_render = st.form_submit_button(
            "Обчислити",
            use_container_width=True,
            help="Перевірте введену функцію" if disabled else None,
            disabled=disabled
        )
    if do_render:
        render_series_info()
