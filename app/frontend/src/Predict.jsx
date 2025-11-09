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
  MenuItem,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip
} from '@mui/material';
import { CloudUpload } from '@mui/icons-material';

function Visualisation() {
  const [file, setFile] = useState(null);
  const [results, setResults] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [model, setModel] = useState('NaiveBayes');

  const handleModelSelect = async (e) => {
    setModel(e.target.value);
  }

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      if (selectedFile.type !== 'text/csv' && !selectedFile.name.endsWith('.csv')) {
        setError('Please upload a CSV file');
        setFile(null);
        return;
      }
      setFile(selectedFile);
      setError('');
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setResults(null);
    
    if (!file) {
      setError('Please select a CSV file first');
      return;
    }

    setLoading(true);

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('model', model);

      const response = await axios.post('http://localhost:8000/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setResults(response.data || 'No response from model');
    } catch (err) {
      console.error(err);
      setError('Error connecting to backend or predicting. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  {/* Function to render results in table format */}
  const renderResultsTable = () => {
    if (!results || !Array.isArray(results.predictions)) return null;

    return (
      <Paper elevation={3} sx={{ p: 3, mt: 3 }}>
        <Typography variant="h4" gutterBottom sx={{ mb: 3 }}>
          Classification Results
        </Typography>
        
        {/* Summary Stats */}
        <Box sx={{ mb: 3, display: 'flex', gap: 2 }}>
          <Chip 
            label={`Valid Messages: ${results.valid_count || 0}`} 
            color="success" 
            variant="outlined" 
          />
          <Chip 
            label={`Spam: ${results.spam_count || 0}`} 
            color="error" 
            variant="outlined" 
          />
        </Box>

        {/* Results Table */}
        <TableContainer component={Paper} variant="outlined">
          <Table sx={{ minWidth: 650 }} aria-label="classification results">
            <TableHead>
              <TableRow sx={{ backgroundColor: 'primary.main' }}>
                <TableCell sx={{ color: 'white', fontWeight: 'bold' }}>Message Title</TableCell>
                <TableCell sx={{ color: 'white', fontWeight: 'bold' }}>Message Body</TableCell>
                <TableCell sx={{ color: 'white', fontWeight: 'bold' }}>Classification</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {results.predictions.map((row, index) => (
                <TableRow 
                  key={index}
                  sx={{ 
                    '&:last-child td, &:last-child th': { border: 0 },
                    backgroundColor: row.prediction === 'spam' ? '#ffebee' : 'transparent'
                  }}
                >
                  <TableCell component="th" scope="row">
                    {row.title || `Message ${index + 1}`}
                  </TableCell>
                  <TableCell sx={{ maxWidth: 400, wordWrap: 'break-word' }}>
                    {row.text ? (
                      row.text.length > 100 
                        ? `${row.text.substring(0, 100)}...` 
                        : row.text
                    ) : 'No content'}
                  </TableCell>
                  <TableCell>
                    <Chip 
                      label={row.prediction === 'spam' ? 'SPAM' : 'VALID'} 
                      color={row.prediction === 'spam' ? 'error' : 'success'}
                      variant="filled"
                    />
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </Paper>
    );
  };

  return (
    <Container
      maxWidth={false}
      disableGutters
      sx={{
        px: 4,
        mt: 8,
        mb: 6,
        display: 'flex',              
        justifyContent: 'center',     
      }}
    >
      <Box sx={{ width: '100%', maxWidth: 1200 }}>
        {/* ---------- Header ---------- */}
        <Typography variant="h3" component="h1" gutterBottom>
          Email Spam Classifier
        </Typography>
        <Typography variant="body1" sx={{ mb: 3 }}>
          Upload a CSV file with email data (columns: title, text) to classify messages as spam or not spam.
        </Typography>

        {/* ---------- File Upload Form ---------- */}
        <Paper elevation={3} sx={{ p: 4, mb: 4, maxWidth: 1000 }}>
          <form onSubmit={handleSubmit}>
            <Box sx={{ mb: 3 }}>
              <Button
                variant="outlined"
                component="label"
                startIcon={<CloudUpload />}
              sx={{ mb: 3, width: 200 }}
              >
                Upload CSV File
                <input
                  type="file"
                  hidden
                  accept=".csv"
                  onChange={handleFileChange}
                />
              </Button>
              {file && (
                <Typography variant="body2" sx={{ ml: 1, display: 'inline' }}>
                  Selected: {file.name}
                </Typography>
              )}
              <Typography variant="caption" display="block" sx={{ mt: 1, color: 'text.secondary' }}>
                CSV should have columns: title, text
              </Typography>
            </Box>

            <Select
              label="Model"
              id="modelSelect"
              value={model}
              onChange={handleModelSelect}
              color="primary"
              sx={{ mb: 3, width: 200 }}
            >
              <MenuItem value={"NaiveBayes"}>Naive Bayes</MenuItem>
              <MenuItem value={"LinearRegression"}>Linear Regression</MenuItem>
            </Select>

            <Button
              type="submit"
              variant="contained"
              color="primary"
              size="large"
              disabled={loading || !file}
              fullWidth                 
            >
              {loading ? <CircularProgress size={24} /> : 'Classify Emails'}
            </Button>
          </form>
        </Paper>

        {/* ---------- Error Message ---------- */}
        {error && (
          <Typography color="error" sx={{ mb: 2 }}>
            {error}
          </Typography>
        )}

        {/* ---------- Results Table ---------- */}
        {renderResultsTable()}
      </Box>
    </Container>
  );
}

export default Visualisation;