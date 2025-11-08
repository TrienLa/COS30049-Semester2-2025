// ===========================================
// 🧠 COS30049 Frontend — Unified App.jsx
// Combines navigation (AppBar + Drawer) with
// integrated PredictPage (Model) and other pages
// ===========================================

import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import {
  AppBar, Toolbar, Typography, Container, Button, Box,
  Drawer, List, ListItem, ListItemIcon, ListItemText, IconButton,
  Switch, Snackbar, Alert, Fab, Divider, Paper
} from '@mui/material';
import {
  Menu as MenuIcon,
  Home as HomeIcon,
  Info as InfoIcon,
  Add as AddIcon,
  LightMode as LightModeIcon,
  DarkMode as DarkModeIcon,
  Equalizer as EqualizerIcon
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

const darkTheme = createTheme({
  palette: {
    mode: 'dark',
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
          flexWrap: 'wrap',
          width: '100%',
          maxWidth: 1200,
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
              flex: '1 1 30%',
              maxWidth: 340,
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
        The system focuses on malware classification, using an AI model trained to
  identify patterns and features associated with malicious software. Through the
  website, users can input data, view classification results, and explore
  interactive visualizations that display model confidence and detection
  insights. The application highlights how artificial intelligence can be
  deployed in cybersecurity contexts to improve awareness and digital threat
  detection.
      </Typography>

      {/* ---------- Team Member Cards ---------- */}
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'stretch',
          flexWrap: 'wrap',
          gap: 4,
          width: '100%',
          maxWidth: 1200,
          pb: 6,
        }}
      >
        {[
          { name: 'Santiago Merchant', about: '[About]' },
          { name: 'Thien La', about: '[About]' },
          { name: 'Trien La', about: '[About]' },
        ].map((member, index) => (
          <Paper
            key={index}
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
              {member.name}
            </Typography>
            <Typography variant="body2">{member.about}</Typography>
          </Paper>
        ))}
      </Box>
      {/* End Team Member cards, remove this section if we want to get rid of them */}
    </Container>
  );
}

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
        alignItems: 'center',
      }}
    >
      <Typography variant="h3" gutterBottom>
        Model Visualisations & Insights
      </Typography>
      <Typography variant="body1" sx={{ mb: 4, textAlign: 'center', maxWidth: 900 }}>
        Explore how the spam classifier performs on test data, and view insights
        from the dataset used to train it.
      </Typography>

      {[
        {
          title: 'Dataset Distribution',
          desc: 'Number of spam vs. non-spam emails in the dataset.',
          image: 'spam_distribution.png'
        },
        {
          title: 'Dataset Spam Words Distribution',
          desc: 'Visualisation based on the amount of words appeared in the dataset.',
          image: 'spam_wordcloud.png'
        },
        {
          title: 'Naive Bayes Confusion Matrix',
          desc: 'Visual representation of predicted vs. actual classifications of the NB Model',
          image: 'nb_model.png'
        },
        {
          title: 'Linear Regression Confusion Matrix',
          desc: 'Visual representation of predicted vs. actual classifications of the LR Model',
          image: 'lr_model.png'
        },
      ].map((section, index) => (
        <Paper
          key={index}
          sx={{
            p: 4,
            mb: 5,
            borderRadius: 3,
            maxWidth: 1000,
          }}
        >
          <Typography variant="h5" gutterBottom>
            {section.title}
          </Typography>
          <Typography variant="body2" sx={{ mb: 2 }}>
            {section.desc}
          </Typography>
          <Box
            component="img"
            display="flex"
            justifyContent="center"
            alignItems="center"
            src={section.image}
          />
        </Paper>
      ))}
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
          <ListItemIcon><EqualizerIcon /></ListItemIcon>
          <ListItemText primary="Visualisations" />
        </ListItem>
      </List>
    </Box>
  );

  // ===========================================
  // 🌐 PAGE STRUCTURE + ROUTING
  // ===========================================
  return (
    <Router>
      <ThemeProvider theme={darkMode ? darkTheme : theme}>
        <Box
          sx={{
            display: 'flex',
            flexDirection: 'column',
            minHeight: '100vh',
            width: '100vw',
            bgcolor: darkMode ? 'grey.900' : 'background.default',
            color: darkMode ? 'common.white' : 'common.black',
            overflowX: 'hidden',
            overflowY: 'auto',
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
          <Fab 
          aria-label="add" 
          sx={{ position: 'fixed', bottom: 16, right: 16 }}
          onClick={handleDarkModeToggle}>
            {darkMode ? <DarkModeIcon /> : <LightModeIcon />}
          </Fab>

          {/* ✅ SNACKBAR */}
          <Snackbar open={snackbarOpen} autoHideDuration={6000} onClose={handleSnackbarClose}>
            <Alert onClose={handleSnackbarClose} severity="success" sx={{ width: '100%' }}>
              {darkMode ? 'Dark mode enabled!' : 'Light mode enabled!'}
            </Alert>
          </Snackbar>
        </Box>
      </ThemeProvider>
    </Router>
  );
}

export default App;
