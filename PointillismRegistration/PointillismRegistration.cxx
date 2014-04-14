#include "itkImageFileWriter.h"
#include "itkHoughTransformRadialVotingImageFilter.h"
#include "itkAdditiveGaussianNoiseImageFilter.h"
#include "itkConfidenceConnectedImageFilter.h"
#include "itkCurvatureFlowImageFilter.h"
#include "itkSobelEdgeDetectionImageFilter.h"

#include "itkHessianRecursiveGaussianImageFilter.h"
#include "itkHessianToObjectnessMeasureImageFilter.h"
#include "itkMultiScaleHessianBasedMeasureImageFilter.h"

#include "itkPluginUtilities.h"

#include "PointillismRegistrationCLP.h"

// Use an anonymous namespace to keep class types and function names
// from colliding when module is used as shared object module.  Every
// thing should be in an anonymous namespace except for the module
// entry point, e.g. main()
//
namespace
{

template <class T>
int DoIt( int argc, char * argv[], T )
{
  PARSE_ARGS;

  // Data types
  const   unsigned int    Dimension = 3;
  typedef T               InputPixelType;
  typedef float           InternalPixelType;
  typedef T               OutputPixelType;  
  typedef itk::NumericTraits< InternalPixelType >::RealType RealPixelType;
  typedef itk::SymmetricSecondRankTensor< RealPixelType, Dimension > HessianPixelType;

  // Image types
  typedef itk::Image<InputPixelType,     Dimension>  InputImageType;
  typedef itk::Image<InternalPixelType,  Dimension>  InternalImageType;
  typedef itk::Image<OutputPixelType,    Dimension>  OutputImageType;
  typedef itk::Image<HessianPixelType,   Dimension>  HessianImageType;

  // Reader / Writer
  typedef itk::ImageFileReader<InputImageType>    ReaderType;
  typedef itk::ImageFileWriter<InternalImageType> InternalWriterType;
  typedef itk::ImageFileWriter<OutputImageType>   OutputWriterType;

  // Filters
  typedef itk::AdditiveGaussianNoiseImageFilter<InputImageType, InternalImageType> AddGaussianNoiseFilterType;
  typedef itk::HoughTransformRadialVotingImageFilter<InternalImageType, InternalImageType> HoughTransformFilterType;
  typedef itk::ConfidenceConnectedImageFilter<InternalImageType, InternalImageType> ConfidenceConnectedFilterType; 
  typedef itk::CurvatureFlowImageFilter<InputImageType, InternalImageType> CurvatureFlowFilterType;
  typedef itk::SobelEdgeDetectionImageFilter<InternalImageType, InternalImageType> SobelEdgeFilterType;
  typedef itk::HessianRecursiveGaussianImageFilter<InputImageType, InputImageType> HessianRecursiveGaussianFilterType;
  typedef itk::HessianToObjectnessMeasureImageFilter<HessianImageType, InternalImageType> HessianToObjectnessFilterType;
  typedef itk::MultiScaleHessianBasedMeasureImageFilter< InputImageType, HessianImageType, InternalImageType> MultiScaleEnhancementFilterType;

  // Read input volume
  typename ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName( inputVolume.c_str() );
  reader->Update();

  // Add noise
  /*
  typename AddGaussianNoiseFilterType::Pointer gFilter = AddGaussianNoiseFilterType::New();
  gFilter->SetInput( reader->GetOutput() );
  gFilter->SetNumberOfThreads(1);
  gFilter->SetStandardDeviation(stddev);
  gFilter->Update();
  
  // Smooth image
  typename CurvatureFlowFilterType::Pointer sFilter = CurvatureFlowFilterType::New();
  sFilter->SetInput( reader->GetOutput() );
  sFilter->SetNumberOfIterations( 3 );
  sFilter->SetTimeStep(0.0625);
  sFilter->Update();

  // Intermediate Volume = Noisy image
  typename InternalWriterType::Pointer writerInternal = InternalWriterType::New();
  writerInternal->SetFileName( intermediateVolume.c_str() );
  writerInternal->SetInput( sFilter->GetOutput() );
  writerInternal->SetUseCompression(1);
  writerInternal->Update();

  // Apply Hough Transform
  typename HoughTransformFilterType::Pointer hFilter = HoughTransformFilterType::New();
  hFilter->SetInput( sFilter->GetOutput() );
  hFilter->SetNumberOfSpheres( numberOfSpheres );
  hFilter->SetMinimumRadius( minRadius );
  hFilter->SetMaximumRadius( maxRadius );
  hFilter->SetSigmaGradient( sigmaGrad );
  hFilter->SetVariance( variance );
  hFilter->SetSphereRadiusRatio( sphereRadiusRatio );
  hFilter->SetVotingRadiusRatio( votingRadiusRatio );
  hFilter->SetThreshold( threshold );
  hFilter->SetOutputThreshold( outputThreshold );
  hFilter->SetGradientThreshold( gradThreshold );
  hFilter->Update();

  // Get image informations
  typename InputImageType::DirectionType imageDirection = reader->GetOutput()->GetDirection();
  typename InputImageType::PointType     origin         = reader->GetOutput()->GetOrigin();
  typename InputImageType::SpacingType   spacing        = reader->GetOutput()->GetSpacing();

  // Create csv file with fiducial center positions
  std::ofstream csvFile;
  csvFile.open("/home/snr/HoughFiducials.fcsv", std::ios::out);
  csvFile << "# Markups fiducial file version = 4.3" << std::endl
	  << "# CoordinateSystem = 0" << std::endl
	  << "# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID" << std::endl;

  typedef itk::EllipseSpatialObject<Dimension>::TransformType::OffsetType Coordinate;
  typedef typename HoughTransformFilterType::SpheresListType SpheresListType;
  SpheresListType spheresList = hFilter->GetSpheres();

  std::cerr << "Number of sphere detected: " << spheresList.size() << std::endl
	    << "----------------------------------" << std::endl << std::endl;

  unsigned int count = 1;
  typename SpheresListType::const_iterator itSpheres = spheresList.begin();  
  while(itSpheres != spheresList.end())
    {
    Coordinate offset = (*itSpheres)->GetObjectToParentTransform()->GetOffset();

    // Convert pixel offset to RAS coordinate
    Coordinate PhysicalOffset = offset;
    PhysicalOffset[0] *= spacing[0];
    PhysicalOffset[1] *= spacing[1];
    PhysicalOffset[2] *= spacing[2];
    double RASCenter[3] = { 0,0,0 };
    RASCenter[0] = -origin[0] - (PhysicalOffset[0]*imageDirection[0][0] + 
				 PhysicalOffset[1]*imageDirection[0][1] + 
				 PhysicalOffset[2]*imageDirection[0][2]);
    RASCenter[1] = -origin[1] - (PhysicalOffset[0]*imageDirection[1][0] + 
				 PhysicalOffset[1]*imageDirection[1][1] + 
				 PhysicalOffset[2]*imageDirection[1][2]);
    RASCenter[2] = origin[2] + (PhysicalOffset[0]*imageDirection[2][0] + 
				PhysicalOffset[1]*imageDirection[2][1] + 
				PhysicalOffset[2]*imageDirection[2][2]);
    
    std::cerr << "Fiducial " << count << ": " << std::endl
	      << "  Pixel Offset: " << offset << std::endl
	      << "  Center (RAS): " << RASCenter[0] << "," << RASCenter[1] << "," << RASCenter[2] << std::endl
	      << "  Radius: " << (*itSpheres)->GetRadius()[0] << std::endl
	      << std::endl;

    csvFile << "houghFiducial_" << count << "," 
	    << RASCenter[0] << "," 
	    << RASCenter[1] << "," 
	    << RASCenter[2] << "," 
	    << "0,0,0,1,1,1,0," << count << ",," << std::endl;
    

    // Add seed in confidence connected filter
    ++itSpheres;
    ++count;
    }

  csvFile.close();
  std::cerr << "----------------------------------" << std::endl;
  
  //cFilter->Update();
  */

/*
  typename HessianRecursiveGaussianFilterType::Pointer hessianGaussian =
    HessianRecursiveGaussianFilterType::New();
  hessianGaussian->SetInput( reader->GetOutput() );
  hessianGaussian->SetSigma(sigma);
  hessianGaussian->Update();
*/

  typename HessianToObjectnessFilterType::Pointer hessianObjectness =
    HessianToObjectnessFilterType::New();
  hessianObjectness->SetBrightObject(true);
  hessianObjectness->SetScaleObjectnessMeasure(false);
  hessianObjectness->SetAlpha(alpha);
  hessianObjectness->SetBeta(beta);
  hessianObjectness->SetGamma(gamma);
  hessianObjectness->SetObjectDimension(0);

  typename MultiScaleEnhancementFilterType::Pointer multiScale = MultiScaleEnhancementFilterType::New();
  multiScale->SetInput(reader->GetOutput());
  multiScale->SetSigmaMinimum(minSigma);
  multiScale->SetSigmaMaximum(maxSigma);
  multiScale->SetNumberOfSigmaSteps(stepSigma);
  multiScale->SetHessianToMeasureFilter(hessianObjectness);
  multiScale->Update();

  // Apply Hough Transform
  typename HoughTransformFilterType::Pointer hFilter = HoughTransformFilterType::New();
  hFilter->SetInput( multiScale->GetOutput() );
  hFilter->SetNumberOfSpheres( numberOfSpheres );
  hFilter->SetMinimumRadius( minRadius );
  hFilter->SetMaximumRadius( maxRadius );
  hFilter->SetSigmaGradient( sigmaGrad );
  hFilter->SetVariance( variance );
  hFilter->SetSphereRadiusRatio( sphereRadiusRatio );
  hFilter->SetVotingRadiusRatio( votingRadiusRatio );
  hFilter->SetThreshold( threshold );
  hFilter->SetOutputThreshold( outputThreshold );
  hFilter->SetGradientThreshold( gradThreshold );
  hFilter->Update();

  // Get image informations
  typename InputImageType::DirectionType imageDirection = reader->GetOutput()->GetDirection();
  typename InputImageType::PointType     origin         = reader->GetOutput()->GetOrigin();
  typename InputImageType::SpacingType   spacing        = reader->GetOutput()->GetSpacing();

  // Create csv file with fiducial center positions
  std::ofstream csvFile;
  csvFile.open("/home/snr/HoughFiducials.fcsv", std::ios::out);
  csvFile << "# Markups fiducial file version = 4.3" << std::endl
	  << "# CoordinateSystem = 0" << std::endl
	  << "# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID" << std::endl;

  typedef itk::EllipseSpatialObject<Dimension>::TransformType::OffsetType Coordinate;
  typedef typename HoughTransformFilterType::SpheresListType SpheresListType;
  SpheresListType spheresList = hFilter->GetSpheres();

  std::cerr << "Number of sphere detected: " << spheresList.size() << std::endl
	    << "----------------------------------" << std::endl << std::endl;

  unsigned int count = 1;
  typename SpheresListType::const_iterator itSpheres = spheresList.begin();  
  while(itSpheres != spheresList.end())
    {
    Coordinate offset = (*itSpheres)->GetObjectToParentTransform()->GetOffset();

    // Convert pixel offset to RAS coordinate
    Coordinate PhysicalOffset = offset;
    PhysicalOffset[0] *= spacing[0];
    PhysicalOffset[1] *= spacing[1];
    PhysicalOffset[2] *= spacing[2];
    double RASCenter[3] = { 0,0,0 };
    RASCenter[0] = -origin[0] - (PhysicalOffset[0]*imageDirection[0][0] + 
				 PhysicalOffset[1]*imageDirection[0][1] + 
				 PhysicalOffset[2]*imageDirection[0][2]);
    RASCenter[1] = -origin[1] - (PhysicalOffset[0]*imageDirection[1][0] + 
				 PhysicalOffset[1]*imageDirection[1][1] + 
				 PhysicalOffset[2]*imageDirection[1][2]);
    RASCenter[2] = origin[2] + (PhysicalOffset[0]*imageDirection[2][0] + 
				PhysicalOffset[1]*imageDirection[2][1] + 
				PhysicalOffset[2]*imageDirection[2][2]);
    
    std::cerr << "Fiducial " << count << ": " << std::endl
	      << "  Pixel Offset: " << offset << std::endl
	      << "  Center (RAS): " << RASCenter[0] << "," << RASCenter[1] << "," << RASCenter[2] << std::endl
	      << "  Radius: " << (*itSpheres)->GetRadius()[0] << std::endl
	      << std::endl;

    csvFile << "houghFiducial_" << count << "," 
	    << RASCenter[0] << "," 
	    << RASCenter[1] << "," 
	    << RASCenter[2] << "," 
	    << "0,0,0,1,1,1,0," << count << ",," << std::endl;
    

    // Add seed in confidence connected filter
    ++itSpheres;
    ++count;
    }

  csvFile.close();
  std::cerr << "----------------------------------" << std::endl;
  

  // Write output volume
  typename InternalWriterType::Pointer writerInternal = InternalWriterType::New();
  writerInternal->SetFileName( intermediateVolume.c_str() );
  writerInternal->SetInput( hFilter->GetOutput() );
  writerInternal->SetUseCompression(1);
  writerInternal->Update();

  return EXIT_SUCCESS;
}

} // end of anonymous namespace

int main( int argc, char * argv[] )
{
  PARSE_ARGS;

  itk::ImageIOBase::IOPixelType     pixelType;
  itk::ImageIOBase::IOComponentType componentType;

  try
    {
    itk::GetImageType(inputVolume, pixelType, componentType);

    // This filter handles all types on input, but only produces
    // signed types
    switch( componentType )
      {
      case itk::ImageIOBase::UCHAR:
        return DoIt( argc, argv, static_cast<unsigned char>(0) );
        break;
      case itk::ImageIOBase::CHAR:
        return DoIt( argc, argv, static_cast<char>(0) );
        break;
      case itk::ImageIOBase::USHORT:
        return DoIt( argc, argv, static_cast<unsigned short>(0) );
        break;
      case itk::ImageIOBase::SHORT:
        return DoIt( argc, argv, static_cast<short>(0) );
        break;
      case itk::ImageIOBase::UINT:
        return DoIt( argc, argv, static_cast<unsigned int>(0) );
        break;
      case itk::ImageIOBase::INT:
        return DoIt( argc, argv, static_cast<int>(0) );
        break;
      case itk::ImageIOBase::ULONG:
        return DoIt( argc, argv, static_cast<unsigned long>(0) );
        break;
      case itk::ImageIOBase::LONG:
        return DoIt( argc, argv, static_cast<long>(0) );
        break;
      case itk::ImageIOBase::FLOAT:
        return DoIt( argc, argv, static_cast<float>(0) );
        break;
      case itk::ImageIOBase::DOUBLE:
        return DoIt( argc, argv, static_cast<double>(0) );
        break;
      case itk::ImageIOBase::UNKNOWNCOMPONENTTYPE:
      default:
        std::cout << "unknown component type" << std::endl;
        break;
      }
    }

  catch( itk::ExceptionObject & excep )
    {
    std::cerr << argv[0] << ": exception caught !" << std::endl;
    std::cerr << excep << std::endl;
    return EXIT_FAILURE;
    }
  return EXIT_SUCCESS;
}



