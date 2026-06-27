import axios from "axios";

const BACKEND_URL =
  process.env.REACT_APP_BACKEND_URL || "http://127.0.0.1:8000";

export const API_BASE = `${BACKEND_URL}/api`;

const api = axios.create({
  baseURL: API_BASE,
  withCredentials: true,
});

export function formatApiError(error) {
  if (error?.response?.data?.detail) {
    return error.response.data.detail;
  }

  if (error?.response?.data?.message) {
    return error.response.data.message;
  }

  if (error?.message) {
    return error.message;
  }

  return "Something went wrong.";
}

export default api;