# causality-probe (español)

**Detecta el sesgo de anticipación (*look-ahead*) en las variables de un backtest, con controles positivos.**

Una variable es causal en el instante *t* solo si depende de los datos disponibles hasta *t*:

```
construir(historia_completa).loc[t]  ==  construir(historia_hasta_t).iloc[-1]
```

La sonda comprueba esa igualdad columna por columna en puntos al azar y en cada cambio de día,
semana y mes, que es donde viven los errores de remuestreo. No hace falta leer el código: si hay
diferencia, hay futuro dentro de la variable.

Una sonda que no encuentra nada no demuestra nada si no se prueba que detecta fugas conocidas.
Por eso, antes de dar un veredicto limpio, `probe()` ejecuta 11 controles:
- **8 fugas conocidas** que tiene que detectar;
- **3 versiones selladas** que tiene que dejar limpias.

Si un control falla, el veredicto es `INVALID_CONTROL_FAILED`.

## Por qué existe

En la investigación del propio autor, una regla de reversión a la media superó el walk-forward, la
revisión adversarial y un holdout sellado de 2.768 operaciones. Dio +0,672 R por operación, con
factor de beneficio 2,23 y p ≈ 0,002, y se llegó a operar en una cuenta de evaluación.

La ventaja la fabricaba una sola línea. El cierre diario se calculaba con `resample('1D').last()`,
que etiqueta cada día en su **inicio**, y se propagaba a las velas de 15 minutos. A las 03:00 el
modelo ya conocía el cierre de las 23:45. Con la fuga corregida, la misma regla perdía −4.380 R en
22 años.

Todas las validaciones usaban la misma variable contaminada. La sonda lo detecta en segundos.
Más detalle en [docs/mechanisms.md](docs/mechanisms.md).

Uso, controles y límites: ver el [README en inglés](README.md).

*Software educativo y de investigación. No es asesoría de inversión.*
