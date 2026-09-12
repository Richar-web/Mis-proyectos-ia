import express from "express";
const app=express();

app.get("/senati",(req,res)=>res.send("hola mundo"));
app.get("/usuarios",(req,res)=>res.send("resultado de usuarios"));
app.listen(3000);
console.log("ya funciona")
