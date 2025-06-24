import React, { useState } from "react";
import axios from "axios";
import { Container, Button, Form, Alert, Card } from "react-bootstrap";

const Upload: React.FC = () => {
    const [file, setFile] = useState<File | null>(null);
    const [message, setMessage] = useState("");

    const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        if (event.target.files && event.target.files.length > 0) {
            setFile(event.target.files[0]);
        }
    };

    const handleUpload = async () => {
        if (!file) {
            alert("Please select a file first.");
            return;
        }

        const formData = new FormData();
        formData.append("file", file);

        try {
            const response = await axios.post("http://127.0.0.1:5000/upload", formData, {
                headers: {
                    "Content-Type": "multipart/form-data"
                }
            });
            setMessage(response.data.message || "File uploaded successfully!");
        } catch (error) {
            console.error("Error uploading file:", error);
            setMessage("Upload failed!");
        }
    };

    return (
        <div
            style={{
                minHeight: "100vh",
                display: "flex",
                justifyContent: "center",
                alignItems: "center",
                background: "linear-gradient(135deg, #667eea, #764ba2)",
            }}
        >
            <Container className="d-flex flex-column align-items-center">
                <Card
                    className="p-4 shadow-lg"
                    style={{
                        width: "32rem",
                        borderRadius: "20px",
                        background: "rgba(255, 255, 255, 0.2)",
                        backdropFilter: "blur(10px)",
                        boxShadow: "0px 4px 15px rgba(0, 0, 0, 0.2)",
                    }}
                >
                    <Card.Body>
                        <h1 className="mb-4 text-center" style={{ fontWeight: "bold", color: "#ffffff" }}>
                            Deepfake Detection
                        </h1>
                        <Form>
                            <Form.Group controlId="fileUpload" className="mb-3">
                                <Form.Label style={{ fontWeight: "600", color: "#ffffff" }}>Select a file</Form.Label>
                                <Form.Control type="file" onChange={handleFileChange} />
                            </Form.Group>
                            <Button
                                variant="primary"
                                onClick={handleUpload}
                                className="w-100"
                                style={{
                                    backgroundColor: "#ff7eb3",
                                    borderColor: "#ff7eb3",
                                    fontWeight: "bold",
                                    borderRadius: "10px",
                                    transition: "0.3s",
                                }}
                                onMouseOver={(e) => (e.currentTarget.style.backgroundColor = "#ff4f91")}
                                onMouseOut={(e) => (e.currentTarget.style.backgroundColor = "#ff7eb3")}
                            >
                                Upload
                            </Button>
                        </Form>
                        {message && (
                            <Alert
                                variant="info"
                                className="mt-3 text-center"
                                style={{
                                    background: "rgba(255, 255, 255, 0.3)",
                                    color: "#fff",
                                    borderRadius: "10px",
                                    border: "none",
                                }}
                            >
                                {message}
                            </Alert>
                        )}
                    </Card.Body>
                </Card>
            </Container>
        </div>
    );
};

export default Upload;
