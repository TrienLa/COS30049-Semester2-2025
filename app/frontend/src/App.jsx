import React, { useState } from 'react';
import axios from 'axios';
import { Line } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend } from 'chart.js';
import {
  Container,
  Typography,
  TextField,
  Button,
  Paper,
  Grid,
  Box,
  CircularProgress
} from '@mui/material';
import { ThemeProvider, createTheme } from '@mui/material/styles';

// Registering Chart.js components
ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
  },
});

function App() {
  const [emailData, setEmailData] = useState('');
  const [spamEmail, setSpamEmail] = useState(null);
  const [error, setError] = useState('');
  const [msg, setMsg] = useState('');
  const [chartData, setChartData] = useState(null);
  const [loading, setLoading] = useState(false);

  async function getRootMsg() {

      try {
        await Promise.all(
          axios.get(`http://localhost:8000/`).then((res) => {console.log(res);})
        );

        console.log(msg)

      } catch (err) {
        setError('Something went wrong getting message from root.');
        console.error(err);
      } 
      
  };

  getRootMsg();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setSpamEmail(null);
    setLoading(true);

    try {
      // Axios call to the backend to predict house price
      const response = await axios.get(`http://localhost:8000/predict/${bedrooms}`);
      setSpamEmail(response.data.predicted_result);

      // Prepare data for the chart
      const squareFootages = [1000, 1500, 2000, 2500, 3000];
      const predictions = await Promise.all(
        squareFootages.map(sf =>
          axios.get(`http://localhost:8000/predict/${sf}/${bedrooms}`)
            .then(res => res.data.predicted_result)
        )
      );

      // Creating the chart data using the predictions from the backend
      const newChartData = {
        labels: squareFootages, // X-axis labels (square footage)
        datasets: [
          {
            label: 'Predicted Prices',
            data: predictions,  // Y-axis data (predicted prices)
            borderColor: 'rgb(75, 192, 192)',
            backgroundColor: 'rgba(75, 192, 192, 0.5)',
            tension: 0.1
          },
          {
            label: 'Your Prediction',
            data: [{ x: parseInt(squareFootage), y: response.data.predicted_price }],
            borderColor: 'rgb(255, 99, 132)',
            backgroundColor: 'rgba(255, 99, 132, 0.5)',
            pointRadius: 8,
            pointHoverRadius: 12,
            showLine: false // Show only the point for the user's prediction
          }
        ]
      };
      setChartData(newChartData);  // Set the chart data in state
    } catch (err) {
      setError('Error predicting price. Please try again.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <ThemeProvider theme={theme}>
      <Container>
        <Box>
          
          <Typography variant="h3" component="h1" gutterBottom>
            Email Spam Classifier
          </Typography>

          <Paper elevation={3} sx={{ p: 3, mb: 3 }}>
            <form onSubmit={handleSubmit}>
              <Grid container spacing={2}>

                <Grid size={12}>
                  <TextField
                    fullWidth
                    type="string"
                    label="Email Input"
                    variant="outlined"
                    value={emailData}
                    onChange={(e) => setEmailData(e.target.value)}
                    required
                  />
                </Grid>

                <Grid size={12}>
                  <Button
                    type="submit"
                    variant="contained"
                    color="primary"
                    fullWidth
                    disabled={loading}
                  >
                    {loading ? <CircularProgress size={24} /> : 'Predict Email'}
                  </Button>
                </Grid>

              </Grid>
            </form>

          </Paper>
          {error && (
            <Typography color="error" sx={{ mb: 2 }}>
              {error}
            </Typography>
          )}

          {spamEmail && (
            <Paper elevation={3} sx={{ p: 3 }}>
              <Typography variant="h5" gutterBottom>
                Predicted Price: ${spamEmail.toLocaleString()}
              </Typography>

              {chartData && (
                <Box sx={{ mt: 3 }}>
                  <Line
                    data={chartData}
                    options={{
                      responsive: true,
                      plugins: {
                        legend: {
                          position: 'top',
                        },
                        title: {
                          display: true,
                          text: 'Price Predictions by Square Footage'
                        }
                      },
                      scales: {
                        x: {
                          type: 'linear',
                          position: 'bottom',
                          title: {
                            display: true,
                            text: 'Square Footage'
                          }
                        },
                        y: {
                          title: {
                            display: true,
                            text: 'Predicted Price ($)'
                          }
                        }
                      }
                    }}
                  />
                </Box>
              )}
            </Paper>
          )}
        </Box>
      </Container>
    </ThemeProvider>
  );
}

export default App;
