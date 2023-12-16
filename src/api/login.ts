import {
  type loginBody,
  type loginResponse,
  type registerBody,
} from "types/api";

export const prerender = false;

export async function login(data: any) {
  try {
    // Realizar la solicitud POST
    const response = await fetch("http://127.0.0.1:8000/auth/jwt/login", {
      method: "POST",
      headers: {
        "Content-Type": "application/x-www-form-urlencoded",
      },
      body: data.toString(),
    });

    // Verificar el estado de la respuesta
    if (response.ok) {
      const responseData = await response.json();
      console.log("Respuesta:", responseData);
      const token = responseData.token as loginResponse;

      return token;
    } else {
      console.error("Error en el login:", response.status);
    }
  } catch (error) {
    console.error("Error en la solicitud:", error);
  }
}

export async function signUp(data: registerBody) {
  try {
    // Realizar la solicitud POST
    const response = await fetch("http://127.0.0.1:8000/auth/register", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(data),
    });

    // Verificar el estado de la respuesta
    if (response.ok) {
      console.log("Registro exitoso");
    } else {
      console.error("Error en el registro:", response.status);
    }
  } catch (error) {
    console.error("Error en la solicitud:", error);
  }
}
