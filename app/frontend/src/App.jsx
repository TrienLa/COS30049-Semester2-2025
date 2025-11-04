// ===========================================
// 🧠 COS30049 Frontend — Unified App.jsx
// Combines navigation (AppBar + Drawer) with
// integrated PredictPage (Model) and other pages
// ===========================================

import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import axios from 'axios';
import {
  AppBar, Toolbar, Typography, Container, Grid, Card, CardContent, Button, Box,
  Drawer, List, ListItem, ListItemIcon, ListItemText, IconButton, TextField,
  Switch, Snackbar, Alert, Fab, Dialog, DialogTitle, DialogContent, DialogContentText,
  DialogActions, CircularProgress, LinearProgress, Chip, Avatar, Divider, Paper
} from '@mui/material';
import {
  Menu as MenuIcon,
  Home as HomeIcon,
  Info as InfoIcon,
  Mail as MailIcon,
  Add as AddIcon,
} from '@mui/icons-material';

import PredictPage from './Predict.jsx';
import { ThemeProvider, createTheme } from '@mui/material/styles';

// ===========================================
// 🎨 THEME CONFIGURATION
// ===========================================
const theme = createTheme({
  palette: {
    primary: { main: '#1976d2' },
    secondary: { main: '#dc004e' },
  },
});

// ===========================================
// 🏠 HOME PAGE
// ===========================================
function HomePage() {
  return (
    <Container
      maxWidth={false}
      disableGutters
      sx={{
        px: 4,
        mt: 8,
        mb: 4,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
      }}
    >
      {/* ---------- Header / Hero Section ---------- */}
      <Typography variant="h2" component="h1" gutterBottom>
        Email Spam Classifier
      </Typography>
      <Typography
        variant="h6"
        sx={{ mb: 5, textAlign: 'center', maxWidth: 900 }}
      >
        An AI-powered web app that detects whether an email is spam or legitimate using
        machine learning. Built with FastAPI (Python) and React (JavaScript).
      </Typography>

      {/* ---------- How It Works ---------- */}
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'space-evenly',
          alignItems: 'stretch',
          flexWrap: 'nowrap',          // ✅ stays on one row
          width: '100%',
          maxWidth: 1200,              // ✅ constrains total width to typical viewport
          mb: 6,
          gap: 3,
        }}
      >
        {[
          { step: '1️⃣', text: 'Paste an email message into the classifier.' },
          { step: '2️⃣', text: 'Our AI model analyses the content for spam indicators.' },
          { step: '3️⃣', text: 'Receive an instant prediction: Spam or Not Spam.' },
        ].map((item, index) => (
          <Paper
            key={index}
            elevation={4}
            sx={{
              flex: '1 1 30%',         // ✅ roughly one-third each
              maxWidth: 340,           // ✅ smaller than before — fits within 1080 px width
              minWidth: 280,
              textAlign: 'center',
              p: 3,
              borderRadius: 3,
            }}
          >
            <Typography variant="h4" gutterBottom>
              {item.step}
            </Typography>
            <Typography variant="body1">{item.text}</Typography>
          </Paper>
        ))}
      </Box>


      {/* ---------- Call to Action ---------- */}
      <Button
        variant="contained"
        color="primary"
        size="large"
        component={Link}
        to="/predict"
      >
        Try the Classifier
      </Button>
    </Container>
  );
}


// ===========================================
// 📄 ABOUT PAGE
// ===========================================
function About() {
  return (
    <Container
      maxWidth={false}
      disableGutters
      sx={{
        px: 4,
        mt: 8,
        mb: 4,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
      }}
    >
      {/* ---------- Header ---------- */}
      <Typography variant="h3" gutterBottom>
        About Us
      </Typography>
      <Typography
        variant="body1"
        sx={{ mb: 5, textAlign: 'center', maxWidth: 900 }}
      >
        This project is part of COS30049 - Artificial Intelligence, showcasing how 
        machine learning models can be integrated into modern web apps using 
        FastAPI (Python) and React (JavaScript).
      </Typography>
      <Typography
        variant="body1"
        sx={{ mb: 5, textAlign: 'left', maxWidth: 900 }}
      >
        [this is where we can talk deeper about how everything works in a more technical sense]
      </Typography>

      {/* ---------- Team Member Cards ---------- */}
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'stretch',
          flexWrap: 'wrap',     // ✅ wraps on smaller screens
          gap: 4,               // ✅ spacing between cards
          width: '100%',
          maxWidth: 1200,       // ✅ keeps layout centered
          pb: 6,                // ✅ prevents footer overlap
        }}
      >
        {/* Member 1 */}
        <Paper
          elevation={4}
          sx={{
            p: 3,
            flex: '1 1 300px',
            maxWidth: 360,
            textAlign: 'center',
            borderRadius: 3,
          }}
        >
          <Typography variant="h5" gutterBottom>
            Santiago Merchant
          </Typography>
          <Typography variant="body2">
            [About]
          </Typography>
        </Paper>

        {/* Member 2 */}
        <Paper
          elevation={4}
          sx={{
            p: 3,
            flex: '1 1 300px',
            maxWidth: 360,
            textAlign: 'center',
            borderRadius: 3,
          }}
        >
          <Typography variant="h5" gutterBottom>
            Thien La
          </Typography>
          <Typography variant="body2">
            [About]
          </Typography>
        </Paper>

        {/* Member 3 */}
        <Paper
          elevation={4}
          sx={{
            p: 3,
            flex: '1 1 300px',
            maxWidth: 360,
            textAlign: 'center',
            borderRadius: 3,
          }}
        >
          <Typography variant="h5" gutterBottom>
            Trien La
          </Typography>
          <Typography variant="body2">
            [About]
          </Typography>
        </Paper>
      </Box>
    </Container>
  );
}




// ===========================================
// 📊 DATA VISUALISATIONS PAGE (Placeholder)
// ===========================================
// ===========================================
// 📊 DATA VISUALISATIONS PAGE
// ===========================================
function Visualisations() {
  return (
    <Container
      maxWidth={false}
      disableGutters
      sx={{
        px: 4,
        mt: 8,
        mb: 6,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center', // ✅ keeps everything centered
      }}
    >
      <Typography variant="h3" gutterBottom>
        Model Visualisations & Insights
      </Typography>

      <Typography variant="body1" sx={{ mb: 4, textAlign: 'center', maxWidth: 900 }}>
        Explore how the spam classifier performs on test data, and view insights
        from the dataset used to train it.
      </Typography>

      {/* ---------- Dataset Overview ---------- */}
      <Paper
        sx={{
          p: 4,
          mb: 5,
          borderRadius: 3,
          width: '100%',
          maxWidth: 1000, // ✅ gives room for larger visuals
        }}
      >
        <Typography variant="h5" gutterBottom>
          Dataset Distribution
        </Typography>
        <Typography variant="body2" sx={{ mb: 2 }}>
          Number of spam vs. non-spam emails in the dataset.
        </Typography>
        <Box
          sx={{
            height: 350,
            backgroundColor: 'rgba(0,0,0,0.05)',
            borderRadius: 2,
          }}
        />
      </Paper>

      {/* ---------- Model Performance ---------- */}
      <Paper
        sx={{
          p: 4,
          mb: 5,
          borderRadius: 3,
          width: '100%',
          maxWidth: 1000,
        }}
      >
        <Typography variant="h5" gutterBottom>
          Model Accuracy
        </Typography>
        <Typography variant="body2" sx={{ mb: 2 }}>
          Comparison of training and validation accuracy across epochs.
        </Typography>
        <Box
          sx={{
            height: 350,
            backgroundColor: 'rgba(0,0,0,0.05)',
            borderRadius: 2,
          }}
        />
      </Paper>

      {/* ---------- Confusion Matrix ---------- */}
      <Paper
        sx={{
          p: 4,
          mb: 5,
          borderRadius: 3,
          width: '100%',
          maxWidth: 1000,
        }}
      >
        <Typography variant="h5" gutterBottom>
          Confusion Matrix
        </Typography>
        <Typography variant="body2" sx={{ mb: 2 }}>
          Visual representation of predicted vs. actual classifications.
        </Typography>
        <Box
          sx={{
            height: 350,
            backgroundColor: 'rgba(0,0,0,0.05)',
            borderRadius: 2,
          }}
        />
      </Paper>
    </Container>
  );
}


// ===========================================
// 🧩 MAIN APP COMPONENT
// ===========================================
function App() {
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [darkMode, setDarkMode] = useState(false);
  const [snackbarOpen, setSnackbarOpen] = useState(false);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [loading, setLoading] = useState(false);

  const toggleDrawer = (open) => (event) => {
    if (event.type === 'keydown' && (event.key === 'Tab' || event.key === 'Shift')) return;
    setDrawerOpen(open);
  };

  const handleDarkModeToggle = () => {
    setDarkMode(!darkMode);
    setSnackbarOpen(true);
  };

  const handleSnackbarClose = (event, reason) => {
    if (reason === 'clickaway') return;
    setSnackbarOpen(false);
  };

  const handleDialogOpen = () => setDialogOpen(true);
  const handleDialogClose = () => setDialogOpen(false);

  const handleSubmit = () => {
    setLoading(true);
    setTimeout(() => {
      setLoading(false);
      handleDialogClose();
      setSnackbarOpen(true);
    }, 2000);
  };

  // ===========================================
  // 🧭 SIDE DRAWER MENU
  // ===========================================
  const drawerContent = (
    <Box
      sx={{ width: 250 }}
      role="presentation"
      onClick={toggleDrawer(false)}
      onKeyDown={toggleDrawer(false)}
    >
      <List>
        <ListItem button component={Link} to="/">
          <ListItemIcon><HomeIcon /></ListItemIcon>
          <ListItemText primary="Home" />
        </ListItem>

        <ListItem button component={Link} to="/about">
          <ListItemIcon><InfoIcon /></ListItemIcon>
          <ListItemText primary="About" />
        </ListItem>

        <ListItem button component={Link} to="/predict">
          <ListItemIcon><AddIcon /></ListItemIcon>
          <ListItemText primary="Predict" />
        </ListItem>

        <ListItem button component={Link} to="/visualisations">
          <ListItemIcon><InfoIcon /></ListItemIcon>
          <ListItemText primary="Visualisations" />
        </ListItem>

        <ListItem button onClick={handleDialogOpen}>
          <ListItemIcon><MailIcon /></ListItemIcon>
          <ListItemText primary="Contact" />
        </ListItem>
      </List>

      <Divider />

      <List>
        <ListItem>
          <ListItemText primary="Dark Mode" />
          <Switch checked={darkMode} onChange={handleDarkModeToggle} />
        </ListItem>
      </List>
    </Box>
  );

  // ===========================================
  // 🌐 PAGE STRUCTURE + ROUTING
  // ===========================================
  return (
    <Router>
      <ThemeProvider theme={theme}>
        <Box
          sx={{
            display: 'flex',
            flexDirection: 'column',
            minHeight: '100vh',
            width: '100vw',               // ✅ ensures it matches viewport width
            bgcolor: darkMode ? 'grey.900' : 'background.default',
            color: darkMode ? 'common.white' : 'common.black',
            overflowX: 'hidden',
            overflowY: 'auto',            // ✅ keeps scrolling behaviour clean
          }}
        >

        
          {/* 🔝 APP BAR */}
          <AppBar position="static" sx={{ width: '100%' }}>
            <Toolbar>
              <IconButton edge="start" color="inherit" onClick={toggleDrawer(true)}>
                <MenuIcon />
              </IconButton>

              <Typography variant="h6" sx={{ flexGrow: 1 }}>
                Email Spam Classifier
              </Typography>

              <Button color="inherit" component={Link} to="/">Home</Button>
              <Button color="inherit" component={Link} to="/about">About</Button>
              <Button color="inherit" component={Link} to="/predict">Predict</Button>
              <Button color="inherit" component={Link} to="/visualisations">Visualisations</Button>
              <Button color="inherit" onClick={handleDialogOpen}>Contact</Button>
            </Toolbar>
          </AppBar>

          {/* 📂 SIDE DRAWER */}
          <Drawer anchor="left" open={drawerOpen} onClose={toggleDrawer(false)}>
            {drawerContent}
          </Drawer>

          {/* 🧭 ROUTES */}
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/about" element={<About />} />
            <Route path="/predict" element={<PredictPage />} />
            <Route path="/visualisations" element={<Visualisations />} />
          </Routes>

          {/* 🦶 FOOTER */}
          <Box component="footer" sx={{ bgcolor: darkMode ? 'grey.800' : 'background.paper', py: 6, mt: 'auto' }}>
            <Container maxWidth="lg">
              <Typography variant="body1">Email Spam Classifier — COS30049</Typography>
              <Typography variant="body2" color="text.secondary">
                {'Copyright © '}
                {new Date().getFullYear()}
                {'.'}
              </Typography>
            </Container>
          </Box>

          {/* ➕ FLOATING ACTION BUTTON */}
          <Fab color="primary" aria-label="add" sx={{ position: 'fixed', bottom: 16, right: 16 }}>
            <AddIcon />
          </Fab>

          {/* ✅ SNACKBAR */}
          <Snackbar open={snackbarOpen} autoHideDuration={6000} onClose={handleSnackbarClose}>
            <Alert onClose={handleSnackbarClose} severity="success" sx={{ width: '100%' }}>
              {darkMode ? 'Dark mode enabled!' : 'Light mode enabled!'}
            </Alert>
          </Snackbar>

          {/* 📩 CONTACT DIALOG */}
          <Dialog open={dialogOpen} onClose={handleDialogClose}>
            <DialogTitle>Contact Us</DialogTitle>
            <DialogContent>
              <DialogContentText>
                Fill out this form to get in touch with the team.
              </DialogContentText>
              <TextField autoFocus margin="dense" label="Your Name" type="text" fullWidth variant="standard" />
              <TextField margin="dense" label="Email Address" type="email" fullWidth variant="standard" />
            </DialogContent>
            <DialogActions>
              <Button onClick={handleDialogClose}>Cancel</Button>
              <Button onClick={handleSubmit} disabled={loading}>
                {loading ? <CircularProgress size={24} /> : 'Submit'}
              </Button>
            </DialogActions>
          </Dialog>
        </Box>
      </ThemeProvider>
    </Router>
  );
}

export default App;
