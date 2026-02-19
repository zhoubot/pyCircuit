/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* TypeDef Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_TYPEDEF_CLASSES
#undef GET_TYPEDEF_CLASSES


namespace mlir {
class AsmParser;
class AsmPrinter;
} // namespace mlir
namespace pyc {

/// Clock signal type
class ClockType;
/// Reset signal type
class ResetType;
class ClockType : public ::mlir::Type::TypeBase<ClockType, ::mlir::Type, ::mlir::TypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "pyc.clock";
  static constexpr ::llvm::StringLiteral dialectName = "pyc";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"clock"};
  }

};
class ResetType : public ::mlir::Type::TypeBase<ResetType, ::mlir::Type, ::mlir::TypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "pyc.reset";
  static constexpr ::llvm::StringLiteral dialectName = "pyc";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"reset"};
  }

};

} // namespace pyc
MLIR_DECLARE_EXPLICIT_TYPE_ID(::pyc::ClockType)
MLIR_DECLARE_EXPLICIT_TYPE_ID(::pyc::ResetType)

#endif // GET_TYPEDEF_CLASSES

