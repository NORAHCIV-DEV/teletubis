document.getElementById('predict-form').addEventListener('submit', async (e) => {
    e.preventDefault();

    // Cambiar estado del botón
    const btn = document.querySelector('.btn-predict');
    const originalText = btn.innerText;
    btn.innerText = 'Calculando...';
    btn.disabled = true;

    // Recolectar datos
    const data = {
        precipitacion_24h_mm: parseFloat(document.getElementById('prec_24').value),
        precipitacion_3d_mm: parseFloat(document.getElementById('prec_3d').value),
        precipitacion_7d_mm: parseFloat(document.getElementById('prec_7d').value),
        pendiente_grados: parseFloat(document.getElementById('pendiente').value),
        elevacion_m: parseFloat(document.getElementById('elevacion').value),
        humedad_pct: parseFloat(document.getElementById('humedad').value),
        departamento: document.getElementById('departamento').value,
        tipo_suelo: document.getElementById('tipo_suelo').value,
        geologia: document.getElementById('geologia').value
    };

    try {
        const response = await fetch('/predecir', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });

        if (!response.ok) throw new Error('Error en la API');

        const result = await response.json();
        
        // Actualizar UI
        updateGauge(result.indice, result.color);
        updateAlertBox(result);

    } catch (error) {
        console.error(error);
        alert('Ocurrió un error al contactar el servidor. Asegúrate de que FastAPI esté corriendo.');
    } finally {
        btn.innerText = originalText;
        btn.disabled = false;
    }
});

function updateGauge(score, color) {
    const fill = document.getElementById('gauge-fill');
    const scoreText = document.getElementById('risk-score');
    
    // Animar texto (contador)
    let currentScore = parseInt(scoreText.innerText) || 0;
    const targetScore = Math.round(score);
    
    // Circunferencia del semicírculo es pi * r (r=40) ≈ 125.6
    // Offset va de 125.6 (0%) a 0 (100%)
    const maxDash = 125.6;
    const offset = maxDash - (maxDash * (score / 100));
    
    fill.style.strokeDashoffset = offset;
    fill.style.stroke = color;

    // Animar el contador de números
    const duration = 1000;
    const steps = 30;
    const stepTime = duration / steps;
    const diff = targetScore - currentScore;
    
    let step = 0;
    const interval = setInterval(() => {
        step++;
        scoreText.innerText = Math.round(currentScore + (diff * (step / steps)));
        if (step >= steps) {
            clearInterval(interval);
            scoreText.innerText = targetScore;
        }
    }, stepTime);
}

function updateAlertBox(result) {
    const box = document.getElementById('alert-box');
    const title = document.getElementById('alert-level');
    const action = document.getElementById('alert-action');

    box.style.borderColor = result.color;
    box.style.boxShadow = `0 0 20px ${result.color}40`;
    
    title.innerText = `Nivel: ${result.nivel}`;
    title.style.color = result.color;
    
    action.innerHTML = `<strong>Alerta:</strong> ${result.alerta}<br><br><strong>Acción Recomendada:</strong> ${result.accion}`;
}
