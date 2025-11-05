import React, { useState } from 'react';
import axios from 'axios';
import {
  Container,
  Typography,
  TextField,
  Button,
  Paper,
  Grid,
  Box,
  CircularProgress,
  Select,
  MenuItem
} from '@mui/material';

function Visualisation() {
  const [emailData, setEmailData] = useState('');
  const [spamResult, setSpamResult] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [model, setModel] = useState('NaiveBayes');

  const handleModelSelect = async (e) => {
    setModel(e.target.value);
  }

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setSpamResult(null);
    setLoading(true);

    try {
      const response = await axios.post('http://localhost:8000/predict', {
        text_input: emailData,
        model: model
      });
      setSpamResult(response.data || 'No response from model');
    } catch (err) {
      console.error(err);
      setError('Error connecting to backend or predicting. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container
        maxWidth={false}
        disableGutters
        sx={{
            px: 4,
            mt: 8,
            mb: 6,
            display: 'flex',              // ✅ enables flex layout
            justifyContent: 'center',     // ✅ centers horizontally
        }}
        >
            <Box sx={{ width: '100%', maxWidth: 1000 }}>

      {/* ---------- Header ---------- */}
      <Typography variant="h3" component="h1" gutterBottom>
        Email Spam Classifier
      </Typography>
      <Typography variant="body1" sx={{ mb: 3 }}>
        Paste or type your email text below to classify it as spam or not spam.
      </Typography>

      {/* ---------- Input Form ---------- */}
      <Paper elevation={3} sx={{ p: 4, mb: 4, maxWidth: 1000 }}>
        <form onSubmit={handleSubmit}>
          <TextField
            fullWidth
            multiline
            minRows={10}              // ⬆ makes the box taller
            maxRows={25}
            label="Email Text"
            variant="outlined"
            value={emailData}
            onChange={(e) => setEmailData(e.target.value)}
            required
            sx={{ mb: 3 }}            // ⬇ adds space before the button
            />

            <Select
            label="Model"
            id="modelSelect"
            value={model}
            onChange={handleModelSelect}
            color="primary"
            sx={{ mb: 3 }}
            >
              <MenuItem value={"NaiveBayes"}>Naive Bayes</MenuItem>
              <MenuItem value={"LinearRegression"}>Linear Regression</MenuItem>
            </Select>

            <Button
            type="submit"
            variant="contained"
            color="primary"
            size="large"
            disabled={loading}
            fullWidth                 // ⬆ makes the button the same width as the text box
            >
            {loading ? <CircularProgress size={24} /> : 'Predict'}
            </Button>
        </form>
      </Paper>

      {/* ---------- Error Message ---------- */}
      {error && (
        <Typography color="error" sx={{ mb: 2 }}>
          {error}
        </Typography>
      )}

      {/* ---------- Model Output ---------- */}
      {spamResult && (
        <Paper elevation={3} sx={{ p: 3 }}>
          <Typography variant="h5" gutterBottom>
            Model Prediction:
          </Typography>
          <Typography variant="body1">{spamResult}</Typography>
        </Paper>
      )}
      </Box>

    </Container>
  );
}

export default Visualisation;
