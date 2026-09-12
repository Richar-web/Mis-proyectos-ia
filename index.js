import express from "express";
import sqlite3 from "sqlite3";

const app = express();

app.use(express.json());

const db = new sqlite3.Database("./database.db", (err) => {
    if (err) {
        console.error("Error al conectar con la base de datos:", err.message);
    } else {
        console.log("Conectado a la base de datos SQLite.");
    }
});

db.run(`CREATE TABLE IF NOT EXISTS recolecciones (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fecha_hora TEXT,
    tipo_residuo TEXT,
    cantidad REAL,
    ubicacion TEXT,
    usuario_registro TEXT
)`);

app.get("/senati", (req, res) => {
    res.send("hola mundo");
});

// Ruta específica que coincide exactamente con @POST("/api/recolecciones") de ApiService.java
app.post("/api/recolecciones", (req, res) => {
    console.log("Petición recibida en /api/recolecciones");
    console.log("Datos recibidos:", req.body);

    const fecha_hora = req.body.fecha_hora || req.body.fechaHora;
    const tipo_residuo = req.body.tipo_residuo || req.body.tipoResiduo;
    const cantidad = req.body.cantidad;
    const ubicacion = req.body.ubicacion;
    const usuario_registro = req.body.usuario_registro || req.body.usuarioRegistro;

    const query = `INSERT INTO recolecciones (fecha_hora, tipo_residuo, cantidad, ubicacion, usuario_registro) VALUES (?, ?, ?, ?, ?)`;
    const params = [fecha_hora, tipo_residuo, cantidad, ubicacion, usuario_registro];

    db.run(query, params, function (err) {
        if (err) {
            console.error("Error al insertar en SQLite:", err.message);
            return res.status(500).json({ error: err.message });
        }

        console.log("Recolección guardada en la BD con ID:", this.lastID);
        res.json({
            mensaje: "¡Recolección registrada con éxito en el servidor!",
            id: this.lastID,
            datos: req.body,
        });
    });
});

// Ruta comodín SOLO para depuración: registra cualquier POST que no coincida arriba,
// para que veas en los logs de Render si la app está pegándole a otra ruta.
app.post("*", (req, res) => {
    console.warn("⚠️ POST a ruta no reconocida:", req.path, "body:", req.body);
    res.status(404).json({
        error: "Ruta no reconocida",
        rutaRecibida: req.path,
        rutaEsperada: "/api/recolecciones",
    });
});

// Render asigna el puerto dinámicamente mediante la variable de entorno PORT.
// Si dejas el puerto fijo en 3000, en Render puede no coincidir con lo que su proxy espera.
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Servidor corriendo en el puerto ${PORT}`);
});