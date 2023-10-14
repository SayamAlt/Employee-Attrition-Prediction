# Employee Attrition Project

## Table of Contents

- [Overview](#overview)
- [Context](#context)
  - [Organization Description](#organization-description)
  - [Objectives](#objectives)
- [Prerequisites](#prerequisites)
- [Docker Installation](#docker-installation)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Steps](#steps)
- [Automation Testing](#automation-testing)
- [Model Monitoring and Logging](#model-monitoring-and-logging)
- [CI/CD Pipeline](#ci-/-cd-pipeline)
- [Additional Information](#additional-information)
- [License](#license)

![Employee Attrition](https://www.techfunnel.com/wp-content/uploads/2020/04/employee-attrition.jpg)
![How to deal with high attrition rate?](https://hrforecast.com/wp-content/uploads/2021/01/HRForecast-blog-How-to-reduce-employee-attrition-scaled.jpg)
![Employee retention strategies](https://cdn.sketchbubble.com/pub/media/catalog/product/optimized1/c/a/ca2f81c4b59e4c5e7a17a38336d0b5ff958dcc16dac2a2f8228e6b221b213734/employee-attrition-slide4.png)

## Overview

This project demonstrates how to deploy a Dockerized machine learning model on the Render platform and set up a continuous integration/continuous deployment (CI/CD) pipeline using GitHub Actions. The goal of this project is to predict employee attrition in an organization using a machine learning model.

## Context

HR analytics, also known as people analytics, is a data-driven approach to managing human resources. It involves gathering and analyzing data related to employees, such as recruitment, performance, engagement, and retention, to derive insights and make informed decisions. This project explores the application of HR analytics in a hypothetical organization and showcases its benefits in optimizing workforce management.

### Organization Description:

Let's consider a medium-sized technology company called "TechSolutions Inc." The company specializes in software development and has a diverse workforce across different departments, including engineering, marketing, sales, and customer support.

### Objectives

The main objectives of this project are as follows:

<ol>
    <li>Understand the factors influencing employee attrition and job satisfaction.</li>
    <li>Identify key predictors of employee performance.</li>
    <li>Develop strategies to improve employee engagement and retention.</li>
</ol>

## Prerequisites

- [GitHub Account](https://github.com)
- [Render Account](https://render.com)
- [Docker](https://www.docker.com/get-started)

### Docker Installation

If you don't have Docker installed, you can follow the [official Docker installation guide](https://docs.docker.com/get-docker/) to get it set up on your system.


## Getting Started

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/your-username/employee-attrition-prediction.git
   cd employee-attrition-prediction
   ```

2. Build the Docker image:

    ```bash
        docker build -t your-container-name:version .
    ```

3. Run the Docker image:

    ```bash
        docker run -d -p 8080:80 --name your-container-name
    ```

    <ul>
        <li>-d: Run the container in detached mode.</li>
        <li>-p 8080:80: Map port 8080 on your host to port 80 inside the container.</li>
        <li>--name your-container-name: Assign a name to your container.</li>
        <li>your-container-name: Use the name you specified when building the container.</li>
    </ul>

# Deploying a Dockerized Machine Learning Model to Render

This guide provides step-by-step instructions on deploying a Dockerized machine learning model to Render, a platform for hosting web applications and services. By following this guide, you can make your machine learning model accessible over the internet as a web service.

### Prerequisites

Before getting started, make sure you have the following prerequisites:

- A Render account (Sign up at https://render.com/)
- A machine learning model packaged in a Docker container
- Docker installed on your local development machine

### Steps

1. **Create a New Web Service on Render**

   - Log in to your Render account and click the "Add New" button on the dashboard.
   - Choose "Web Service" as your service type.

2. **Configure Your Web Service**

   - Give your service a name and select the appropriate region.
   - Under "Build and Deploy," choose "Use a Dockerfile" and specify the path to your Dockerfile.
   - Click the "Create Web Service" button.

3. **Configure Environment Variables**

   If your machine learning model requires any environment variables (e.g., API keys or configuration settings), you can configure them under the "Environment" section in the Render dashboard.

4. **Set Up Custom Domains (Optional)**

   If you have a custom domain, you can configure it to point to your Render service. You'll need to set up DNS records and configure SSL if necessary.

5. **Deploy Your Service**

   - Push the "Deploy" button in the Render dashboard.
   - Render will automatically build and deploy your Docker container to a web service.

6. **Access Your Deployed Model**

   Once the deployment is complete, you can access your machine learning model by visiting the provided URL in your Render dashboard.

## Automation Testing

To perform unit testing on the deployed machine learning model, a retrain.py script has been written in which the model is re-trained on the updated dataset (if any changes have been made) and the model performance is optimized and evaluated on the test dataset using comprehensive cross validation and hyperparameter tuning. To assess the model's performance, multiple evaluation metrics such as accuracy, precision, recall, F1 score, and ROC-AUC score were used. 

## Model Monitoring and Logging

A formal and structured approach was adopted to meticulously document the performance evaluation metrics of the model. Leveraging the capabilities of the logger.info function from Python's logging module, we ensured that essential metrics such as accuracy, precision, recall, F1 score and ROC-AUC score were systematically logged. This meticulous record-keeping not only provided real-time insights into the efficacy of the machine learning model but also facilitated a comprehensive assessment of its performance over time.

## CI/CD Pipeline

The CI/CD pipeline is automatically triggered when you push to the main branch. It builds the Docker image, pushes it to Docker Hub, and deploys the model on Render. Configuration details can be found in .github/workflows/main.yaml.

## Additional Resources

- [Render Documentation](https://render.com/docs)
- [Docker Documentation](https://docs.docker.com/)
- [Machine Learning Model Deployment Best Practices](https://www.render.com/blog/machine-learning-deployment-best-practices/)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
