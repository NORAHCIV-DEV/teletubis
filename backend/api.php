<?php
// Permitir solicitudes desde el Frontend en React (CORS)
header("Access-Control-Allow-Origin: *");
header("Access-Control-Allow-Headers: Content-Type");
header("Access-Control-Allow-Methods: POST, OPTIONS");

// Manejar preflight requests
if ($_SERVER['REQUEST_METHOD'] === 'OPTIONS') {
    http_response_code(200);
    exit();
}

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $inputJSON = file_get_contents('php://input');
    if (!$inputJSON) {
        echo json_encode(["error" => "No se recibieron datos"]);
        exit();
    }
    
    // Usamos Base64 para evitar problemas con las comillas dobles en la terminal de Windows
    $base64_data = base64_encode($inputJSON);
    
    // Ruta al script de python y al ejecutable
    $python_path = "python"; 
    $script_path = dirname(__DIR__) . DIRECTORY_SEPARATOR . "predict_b64.py";
    
    // Ejecutar el script usando shell_exec
    $command = escapeshellcmd("$python_path \"$script_path\" \"$base64_data\"");
    $output = shell_exec($command);
    
    // Devolver JSON al frontend
    header('Content-Type: application/json');
    echo $output ? $output : json_encode(["error" => "Error ejecutando el script de Python"]);
}
?>
